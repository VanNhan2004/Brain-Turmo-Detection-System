import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ===== CHATBOT IMPORT =====
from chatbot.chatbot import init_chatbot, ask_chat  

# ===== CẤU HÌNH CNN =====
MODEL_PATH = "models/cnn_model.keras"
LABELS = ["U thần kinh đệm", "U màng não", "Không có khối u", "U tuyến yên"]
LOGO_PATH = "assets/LOGO_FIT_NTTU.png"

# ===== HÀM DỰ ĐOÁN CNN =====
model = load_model(MODEL_PATH)
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
    y_predict = model.predict(x)[0]
    pred_idx = np.argmax(y_predict)
    return LABELS[pred_idx], y_predict[pred_idx] * 100

# ===== KHỞI TẠO CHATBOT =====
retriever, prompt, llm = init_chatbot()

# ===== APP =====
root = tk.Tk()
root.title("ỨNG DỤNG NHẬN DIỆN BỆNH NÃO BẰNG CNN + CHATBOT")
root.geometry("1100x700")
root.configure(bg="#f4f6f7")

# ===== BANNER =====
BANNER_COLOR = "#034d6f"
TITLE_COLOR = "#ffffff"
banner_frame = tk.Frame(root, bg=BANNER_COLOR, height=100)
banner_frame.pack(fill="x")
logo_img = Image.open(LOGO_PATH).resize((90, 90))
logo_photo = ImageTk.PhotoImage(logo_img)
tk.Label(banner_frame, image=logo_photo, bg=BANNER_COLOR).pack(side="left", padx=25, pady=5)
tk.Label(
    banner_frame, 
    text="TRƯỜNG ĐẠI HỌC NGUYỄN TẤT THÀNH\nKHOA CÔNG NGHỆ THÔNG TIN",
    font=("Times New Roman", 18, "bold"), fg=TITLE_COLOR, bg=BANNER_COLOR, justify="center"
).place(relx=0.5, rely=0.5, anchor="center")

# ===== KHUNG CHÍNH =====
main_frame = tk.Frame(root, bg="#f4f6f7")
main_frame.pack(fill="both", expand=True)

# ===== SIDEBAR =====
sidebar = tk.Frame(main_frame, bg="#e0f2f1", width=230, relief="ridge", bd=2)
sidebar.pack(side="left", fill="y", padx=10, pady=10)
tk.Label(sidebar, text="CHỨC NĂNG", bg="#e0f2f1", font=("Times New Roman", 14, "bold"), fg="#004d40").pack(pady=15)

# ===== CONTENT =====
content = tk.Frame(main_frame, bg="white", bd=2, relief="ridge")
content.pack(side="left", fill="both", expand=True, padx=10, pady=10)

pages = {}
def show_page(name):
    for frame in pages.values():
        frame.pack_forget()
    pages[name].pack(fill="both", expand=True)

# ======================
# TRANG NHẬN DIỆN BỆNH NÃO
# ======================
page_cnn = tk.Frame(content, bg="white")
pages["cnn"] = page_cnn

left_frame = tk.Frame(page_cnn, bg="white")
left_frame.pack(side="left", fill="y", padx=20, pady=20)
right_frame = tk.Frame(page_cnn, bg="white")
right_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

form_frame = tk.LabelFrame(left_frame, text="Thông tin bệnh nhân", font=("Times New Roman", 14, "bold"),
                           fg="#004d40", bg="white", padx=20, pady=20)
form_frame.pack(pady=10, fill="x")

def create_label_entry(parent, text, row):
    tk.Label(parent,text=text,bg="white",font=("Times New Roman", 12)).grid(row=row, column=0, sticky="e", pady=10, padx=10)
    entry = tk.Entry(parent,width=32,font=("Times New Roman", 12),bd=2,relief="solid")
    entry.grid(row=row, column=1, pady=10, padx=10 )
    return entry

entry_id = create_label_entry(form_frame, "Mã bệnh nhân:", 0)
entry_name = create_label_entry(form_frame, "Họ và tên:", 1)
entry_age = create_label_entry(form_frame, "Tuổi:", 2)
entry_gender = create_label_entry(form_frame, "Giới tính:", 3)
entry_dob = create_label_entry(form_frame, "Ngày sinh:", 4)

img_label = tk.Label(right_frame,bg="white",bd=2,relief="solid",width=260,height=260)
img_label.pack(pady=20)
result_label = tk.Label(right_frame,text="",font=("Times New Roman", 16, "bold"),fg="#004d40",bg="white",justify="left")
result_label.pack(pady=20)

def show_image(path):
    img = Image.open(path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

def classify_image(path):
    label, confidence = predict_image(path)
    info_text = (
        f"Loại bệnh: {label}\n"
        f"Độ chính xác: {confidence:.2f}%"
    )
    result_label.config(text=info_text, fg="#01100E",font=("Times New Roman", 15, "bold"),justify="left")

def choose_image():
    file_path = filedialog.askopenfilename(title="Chọn ảnh MRI/CT", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        show_image(file_path)
        classify_image(file_path)

def reset_form():
    entry_name.delete(0, tk.END)
    entry_age.delete(0, tk.END)
    entry_gender.delete(0, tk.END)
    entry_dob.delete(0, tk.END)
    img_label.config(image='')
    img_label.image = None
    result_label.config(text='')

button_frame = tk.Frame(left_frame, bg="white")
button_frame.pack(pady=15)
tk.Button(button_frame, text="Chọn ảnh để dự đoán", command=choose_image,
          bg="#0288d1", fg="white", font=("Times New Roman", 13, "bold"),
          padx=20, pady=8, relief="flat", cursor="hand2", activebackground="#0277bd").grid(row=0, column=0, padx=5, pady=5)
tk.Button(button_frame, text="Reset", command=reset_form,
          bg="#e53935", fg="white", font=("Times New Roman", 13, "bold"),
          padx=20, pady=8, relief="flat", cursor="hand2", activebackground="#c62828").grid(row=0, column=1, padx=5, pady=5)

# ======================
# TRANG CHATBOT (CHAT Ở DƯỚI)
# ======================
page_chat = tk.Frame(content, bg="#f4f6f7")
pages["chatbot"] = page_chat

# Khung hiển thị chat
chat_canvas = tk.Canvas(page_chat, bg="#f4f6f7", highlightthickness=0)
chat_scrollbar = tk.Scrollbar(page_chat, orient="vertical", command=chat_canvas.yview)
chat_frame = tk.Frame(chat_canvas, bg="#f4f6f7")

chat_frame.bind("<Configure>", lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox("all")))
chat_canvas.create_window((0, 0), window=chat_frame, anchor="nw")
chat_canvas.configure(yscrollcommand=chat_scrollbar.set)

# Ô nhập và nút gửi (đặt ở dưới)
entry_frame = tk.Frame(page_chat, bg="#f4f6f7")
entry_frame.pack(side="bottom", fill="x", padx=10, pady=10)

user_entry = tk.Entry(entry_frame, font=("Times New Roman", 12), relief="solid", bd=1)
user_entry.pack(side="left", fill="x", expand=True, pady=5, ipady=5)

send_btn = tk.Button(entry_frame, text="Gửi", bg="#0288d1", fg="white",
                     font=("Times New Roman", 10, "bold"), relief="flat", padx=10, cursor="hand2")
send_btn.pack(side="right", padx=5)

# Đặt khung chat ở trên
chat_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=(10, 0))
chat_scrollbar.pack(side="right", fill="y")

# ===== Hiển thị bong bóng chat =====
def add_message(text, sender="bot"):
    bubble = tk.Frame(chat_frame, bg="#f4f6f7", pady=5)

    if sender == "user":
        msg = tk.Label(
            bubble, text=text, bg="#F6F6F6", font=("Times New Roman", 12),
            wraplength=700, justify="left", anchor="e", padx=10, pady=5, bd=1, relief="solid"
        )
        msg.pack(anchor="e", padx=10)
        bubble.pack(anchor="e", pady=5, padx=20, fill="none")  
    else:
        msg = tk.Label(
            bubble, text=text, bg="#FFFFFF", font=("Times New Roman", 12),
            wraplength=700, justify="left", anchor="w", padx=10, pady=5, bd=1, relief="solid"
        )
        msg.pack(anchor="w", padx=10)
        bubble.pack(anchor="w", pady=5, padx=20, fill="none")  

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1)

# ===== Gửi tin nhắn =====
def send_message(event=None):
    question = user_entry.get().strip()
    if not question:
        return
    add_message(question, sender="user")
    user_entry.delete(0, tk.END)
    root.update()
    answer = ask_chat(retriever, prompt, llm, question)
    add_message(answer, sender="bot")

send_btn.config(command=send_message)
user_entry.bind("<Return>", send_message)

# ===== SIDEBAR BUTTONS =====
def styled_button(parent, text, color, page_name):
    return tk.Button(parent, text=text, command=lambda: show_page(page_name),
                     bg=color, fg="white", font=("Times New Roman", 12, "bold"),
                     padx=12, pady=8, relief="flat", activebackground="#004d40", cursor="hand2")

styled_button(sidebar, "Nhận diện bệnh não", "#0288d1", "cnn").pack(fill="x", padx=15, pady=10)
styled_button(sidebar, "Chatbot", "#43a047", "chatbot").pack(fill="x", padx=15, pady=10)

show_page("cnn")

root.mainloop()