import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def apply_filter(img_array, filter_type):
    if filter_type == "原始":
        return img_array
    elif filter_type == "黑白":
        return cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    elif filter_type == "復古":
        # 創建復古效果
        sepia = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])
        return cv2.transform(img_array, sepia)
    elif filter_type == "素描":
        # 轉換為灰度圖
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # 反轉圖片
        inverted = 255 - gray
        # 高斯模糊
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        # 混合模式
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    elif filter_type == "銳化":
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(img_array, -1, kernel)
    elif filter_type == "模糊":
        return cv2.GaussianBlur(img_array, (5, 5), 0)
    elif filter_type == "邊緣檢測":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return img_array

def main():
    st.title("圖片編輯工具")
    st.write("上傳圖片並進行編輯")

    # 上傳圖片
    uploaded_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 讀取圖片
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        original_img = img_array.copy()

        # 編輯選項
        st.subheader("編輯選項")
        
        # 濾鏡選擇
        filter_type = st.selectbox(
            "選擇濾鏡",
            ["原始", "黑白", "復古", "素描", "銳化", "模糊", "邊緣檢測"]
        )
        
        # 應用濾鏡
        img_array = apply_filter(img_array, filter_type)
        
        # RGB 調色
        st.write("RGB 調色")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            red = st.slider("紅色通道", -100, 100, 0)
        with col2:
            green = st.slider("綠色通道", -100, 100, 0)
        with col3:
            blue = st.slider("藍色通道", -100, 100, 0)

        # 應用 RGB 調整
        if red != 0 or green != 0 or blue != 0:
            # 分離通道
            b, g, r = cv2.split(img_array)
            
            # 調整各通道
            r = cv2.add(r, red)
            g = cv2.add(g, green)
            b = cv2.add(b, blue)
            
            # 合併通道
            img_array = cv2.merge([b, g, r])

        # 亮度調整
        brightness = st.slider("亮度", -100, 100, 0)
        if brightness != 0:
            img_array = cv2.convertScaleAbs(img_array, alpha=1, beta=brightness)

        # 對比度調整
        contrast = st.slider("對比度", 0.0, 2.0, 1.0)
        if contrast != 1.0:
            img_array = cv2.convertScaleAbs(img_array, alpha=contrast, beta=0)

        # 並排顯示原始圖片和編輯後的圖片
        st.subheader("圖片預覽")
        col1, col2 = st.columns(2)
        with col1:
            st.write("原始圖片")
            st.image(original_img, use_container_width=True)
        with col2:
            st.write("編輯後的圖片")
            st.image(img_array, use_container_width=True)

        # 下載按鈕
        if st.button("下載編輯後的圖片"):
            edited_image = Image.fromarray(img_array)
            buf = io.BytesIO()
            edited_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="點擊下載",
                data=byte_im,
                file_name="edited_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main() 