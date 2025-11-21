import cv2
import numpy as np
import easyocr
import re
import pandas as pd
from datetime import datetime
import os

# GPU가 없으면 CPU 모드로 실행
print("EasyOCR 모델을 불러오는 중입니다... (잠시 대기)")
reader = easyocr.Reader(['ko', 'en'], gpu=False)
print("모델 로드 완료! 카메라를 켭니다...")

def open_cam(index=0):
    # 윈도우 최적화 (CAP_DSHOW)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")
    
    # [수정 1] 해상도 FHD(1920x1080)로 상향 -> 글자가 훨씬 선명해짐
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 실제 적용된 해상도 확인
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"카메라 해상도 설정: {int(w)}x{int(h)}")
    
    return cap

def order_points(pts):
    # 사각형의 4개 점 순서를 (좌상, 우상, 우하, 좌하)로 정렬
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # 찌그러진 사각형을 반듯한 직사각형으로 펴주는 함수
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg_norm(br - bl)
    widthB = np.linalg_norm(tr - tl)
    heightA = np.linalg_norm(tr - br)
    heightB = np.linalg_norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def find_receipt_contour(image):
    # 영수증 외곽선(사각형) 찾기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 점이 4개이고(사각형), 크기가 일정 이상일 때만 영수증으로 인정
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4, 2), edges
    return None, edges

def to_scanned(img):
    # [수정 2] 이진화(흑백) 대신 '회색조 + 선명하게' 처리
    # EasyOCR은 아예 흑백(Binary)보다 회색(Grayscale)에서 더 잘 읽을 때가 많습니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: 대비(Contrast)를 제한적으로 높여서 글자를 진하게 만듦
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def parse_receipt_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)
    
    # 1. 금액 추출 (숫자와 콤마)
    total_price = "0"
    # '합계', '총액' 뒤에 오는 숫자 우선 검색
    price_pattern = r'(합계|총액|결제금액|금액|Total)[^0-9]*([\d,]+)'
    match = re.search(price_pattern, joined)
    if match:
        total_price = match.group(2).replace(',', '')
    else:
        # 못 찾으면 '원' 자 앞에 있는 숫자 중 가장 큰 것을 금액으로 추정
        possible_prices = re.findall(r'([\d,]+)\s*원', joined)
        nums = []
        for p in possible_prices:
            clean_p = p.replace(',', '')
            if clean_p.isdigit():
                nums.append(int(clean_p))
        if nums:
            total_price = str(max(nums))

    # 2. 날짜 추출 (YYYY-MM-DD 또는 YYYY.MM.DD)
    date = "날짜 정보 없음"
    date_match = re.search(r'(\d{4}[-./]\d{1,2}[-./]\d{1,2})', joined)
    if date_match: 
        date = date_match.group(1)

    # 3. 상호명 (간단히 첫 번째 줄을 상호로 가정하거나, '점'으로 끝나는 단어 찾기)
    store = "상호 미상"
    if lines:
        store = lines[0] # 보통 첫 줄이 가게 이름
        
    return {"store": store, "date": date, "total_amount": total_price}

def ocr_and_export(img, excel_path="receipts.xlsx"):
    print("\n[OCR] 이미지 분석 시작... (화면이 잠시 멈춥니다)")
    
    # OCR 실행
    result = reader.readtext(img, detail=0)
    full_text = "\n".join(result)
    print(f"\n--- [읽은 내용] ---\n{full_text}\n---------------------")
    
    # 파싱 및 저장
    parsed = parse_receipt_text(full_text)
    parsed['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            df = pd.concat([df, pd.DataFrame([parsed])], ignore_index=True)
        else:
            df = pd.DataFrame([parsed])
        
        df.to_excel(excel_path, index=False)
        print(f"✅ 엑셀 저장 성공! -> {parsed}")
    except Exception as e:
        print(f"❌ 엑셀 저장 실패 (파일을 닫고 다시 시도하세요): {e}")

def main():
    cap = open_cam(0)
    print("\n=== [사용법] ===")
    print("1. 영수증을 어두운 배경 위에 놓으세요.")
    print("2. 카메라를 움직여 초록색 네모가 영수증을 감싸게 하세요.")
    print("3. 's' 키: 스캔(펴기) 미리보기 (글자가 선명한지 확인!)")
    print("4. 'o' 키: OCR 인식 및 엑셀 저장")
    print("5. 'q' 키: 종료")
    
    last_scanned = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        disp = frame.copy()
        quad, edges = find_receipt_contour(frame)
        
        # 인식된 영역 그리기
        if quad is not None:
            cv2.polylines(disp, [quad.astype(int)], True, (0, 255, 0), 3)
            
        cv2.imshow("Camera", disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('s'):
            if quad is None:
                print("⚠️ 영수증 윤곽선을 못 찾았습니다. 배경을 더 어둡게 해보세요.")
            else:
                # 투시 변환 및 전처리
                warped = four_point_transform(frame, quad)
                last_scanned = to_scanned(warped)
                cv2.imshow("Scanned Preview", last_scanned)
                print("📸 스캔 완료! 미리보기 창의 글자가 선명한가요? 그렇다면 'o'를 누르세요.")
                
        elif key == ord('o'):
            if last_scanned is not None:
                ocr_and_export(last_scanned)
            else:
                print("⚠️ 먼저 's'를 눌러 스캔을 수행하세요.")
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
