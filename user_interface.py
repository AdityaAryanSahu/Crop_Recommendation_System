from tkinter import *
from tkinter import messagebox
from user_interface_backend import *
from simulator import *
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import os

root=Tk()

root.title("CROP RECOMMENDATION SYSTEM")
root.geometry("515x830")

uploaded_soil_image_path = None
soil_type=None

bg_img = Image.open(r'C:\Users\Lenovo\Pictures\Screenshots\image2.jpg')  
bg_img = bg_img.resize((515, 830))     
bg_photo = ImageTk.PhotoImage(bg_img)

# Set the image as a label
background_label = Label(root, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

label_text=["Nitrogen content (0-100)","Phosphorus content (0-100)","Potassium content (0-100)",
            "Temperature (in deg C)","Humidity (% relative)",
            "Soil pH (0-10)","Rainfall (in mm)"]


j=0
entries=[]
for i in range(0,13,2): 
    label=Label(root,text=label_text[j],anchor="w",bg='#8A9A5B', fg="black",
                width=25,font=("Tahoma", 11))
    label.grid(row=i,column=0)
    e=Entry(root,width=60,borderwidth=3,relief="solid",font=("Tahoma", 11),insertbackground="black")
    e.grid(row=i+1,column=0,columnspan=10,padx=10,pady=10)
    entries.append(e)
    j=j+1
result=Label(root,width=38,bg='#8A9A5B', fg="black")

def upload_and_classify_soil_image():
    global uploaded_soil_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        uploaded_soil_image_path = file_path
        messagebox.showinfo("Info","Upload Successful")
            
# method for after clicking submit
def submit_click():
    try:
        values = [e.get().strip() for e in entries]

        # Check for empty input
        if any(v == '' for v in values):
            messagebox.showerror("Input Error", "Please fill all fields.")
            return

        # Convert all to float
        values = list(map(float, values))

        # Define acceptable ranges
        ranges = [
            (0, 100),   # N
            (0, 100),   # P
            (0, 100),   # K
            (-10, 60),  # Temperature
            (0, 100),   # Humidity
            (0, 10),    # pH
            (0, 500)    # Rainfall
        ]

        # Validate ranges
        for i, (val, (low, high)) in enumerate(zip(values, ranges)):
            if not (low <= val <= high):
                messagebox.showerror("Range Error", f"{label_text[i]} must be between {low} and {high}.")
                return

        # Prediction
        y_pred = getData(*values)
       
        global uploaded_soil_image_path

        # Predict soil type only if image is uploaded
        if uploaded_soil_image_path:
            soil_type = predict_soil_type(uploaded_soil_image_path)
            if soil_type:
                valid_crops = soil_rules().get(soil_type.lower(), [])
                if  y_pred[0].lower() in [c.lower() for c in valid_crops]:
                     # Compatible, proceed normally
                    result.config(
                    text=f"Recommended Crop: { y_pred[0]}\n(Suitable for {soil_type} soil)",
                    font=("Tahoma", 15)
                    )
                else:
                    alternatives = ", ".join(valid_crops[:5])  # show top 5 compatible crops
                    result.config(
                    text=(
                     f"'{y_pred[0]}' may not grow well in {soil_type} soil.\n\n"
                    f" Suggested crops for {soil_type} soil:\n{alternatives}"
                    ),
                    font=("Tahoma", 15)
                    )
                    
        else:
            messagebox.showerror("Missing Soil Type", "Please upload a soil image before predicting.")
            return
        
        result.grid(row=17, column=0, columnspan=50, pady=18)
        
    except ValueError:
        messagebox.showerror("Type Error", "Please enter valid numeric values only.")
 
b1=Button(root,text='Submit',command=submit_click,width=20, height=2)
b1.grid(row=15,column=0)

upload_btn = Button(root, text="Upload Soil Image", command=upload_and_classify_soil_image, width=20, height=2)
upload_btn.grid(row=14, column=0, pady=10)

def clear_entries():
    for i in range(0,7):
         entries[i].delete(0,END)
    result.config(text="")
    result.grid_forget()      
    global soil_type
    soil_type = None


clear_button = Button(root, text="Clear", command=clear_entries,width=20, height=2)
clear_button.grid(row=15,column=7)


# below part for simulation only 
sim_label = Label(root, text="", bg='#8A9A5B', fg="black", font=("Tahoma", 11), width=65, justify="left")
sim_label.grid(row=18, column=0, columnspan=10)

simulation_running = False

def simulate_data():
    global simulation_running
    if not simulation_running:
        return

    data, prediction = get_simulated_prediction()
    data['Recommended_crop']=prediction
    
    
    file_exists = os.path.isfile("simulated_data.csv")

    with open("simulated_data.csv", "a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Recommended_crop"])

        if not file_exists or os.stat("simulated_data.csv").st_size == 0:
            writer.writeheader()

        writer.writerow(data)
        
    sim_label.config(text=(
        f"Simulated Input:\n"
        f"NPK = ({data['N']}, {data['P']}, {data['K']}) | "
        f"Temp = {data['temperature']}°C | Humidity = {data['humidity']}%\n"
        f"pH = {data['ph']} | Rainfall = {data['rainfall']}mm\n"
        f"Recommended Crop: {prediction}"
    ))

    root.after(5000, simulate_data)

def start_simulation():
    global simulation_running
    simulation_running = True
    simulate_data()

def stop_simulation():
    global simulation_running
    simulation_running = False
    sim_label.config(text="Simulation stopped.")

sim_btn = Button(root, text="Start Simulation", command=start_simulation, width=20, height=2)
sim_btn.grid(row=16, column=0, pady=10)

stop_btn = Button(root, text="Stop Simulation", command=stop_simulation, width=20, height=2)
stop_btn.grid(row=16, column=7, pady=10)

root.mainloop()