from tkinter import *
from tkinter import messagebox
from user_interface_backend import *
from simulator import *
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import os
import socket
import json
import threading
import requests


# Pi socket setup
HOST = '10.50.229.189'  # Replace with Raspberry Pi IP address
PORT = 9999

client_socket = None
listener_thread = None
listening = False


root=Tk()

root.title("CROP RECOMMENDATION SYSTEM")
root.geometry("560x880")

uploaded_soil_image_path = None
soil_type=None

bg_img = Image.open(r'C:\Users\Lenovo\Pictures\Screenshots\image2.jpg')  
bg_img = bg_img.resize((560, 880))     
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

city_label = Label(root, text="City Name:", bg='#8A9A5B', fg="black", font=("Tahoma", 11), width=25)
city_label.grid(row=14, column=0)
city_entry = Entry(root, width=60, borderwidth=3, relief="solid", font=("Tahoma", 11), insertbackground="black")
city_entry.grid(row=15, column=0, columnspan=10, padx=10, pady=10)

# this fetches the humidty,temp, rain based on the region/city
def fetch_temp_humid_rain():
    API_KEY = "2YKQHG6CPFZWJH3HAMVPZUD38" # replace your api key
    city = city_entry.get().strip()
    if not city:
        messagebox.showerror("Input Error", "Please enter a city name for rainfall lookup.")
        return
    try:
        lat, lon= get_coordinates(city)
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat}%2C{lon}?unitGroup=metric&key={API_KEY}&contentType=json"
        response = requests.get(url)
        print(url)
        data = response.json()
        today_data = data.get("days", [{}])[0]
        if not today_data:
            messagebox.showwarning("Data error", "No data available, Enter manually")
            
        rain_mm=get_avg_rain(lat, lon)
        print(f"rain returned as: {rain_mm}")    
        entries[6].delete(0, END)
        entries[6].insert(0, str(round(rain_mm, 2)))
        
        humidity= today_data.get("humidity",0)
        entries[4].delete(0, END)
        entries[4].insert(0, str(round(humidity, 2)))
        
        temp= today_data.get("temp",0)
        entries[3].delete(0, END)
        entries[3].insert(0, str(round(temp, 2)))
        
        messagebox.showinfo("API Update", f"For {city}, temp, humidity, rainfall updated")
    except Exception as e:
        messagebox.showerror("API Error", f"Failed to fetch rainfall: {e}")

def upload_and_classify_soil_image():
    global uploaded_soil_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        uploaded_soil_image_path = file_path
        messagebox.showinfo("Info","Upload Successful")
            
# method for after clicking submit
def submit_click():
    try:
        fetch_temp_humid_rain()
    except Exception as e:
        messagebox.showerror("fetch error",  "fetching weather data failed")
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
                if  y_pred.lower() in [c.lower() for c in valid_crops]:
                     # Compatible, proceed normally
                    result.config(
                    text=f"Recommended Crop: { y_pred}\n(Suitable for {soil_type} soil)",
                    font=("Tahoma", 15)
                    )
                else:
                    alternatives = ", ".join(valid_crops[:5])  # show top 5 compatible crops
                    result.config(
                    text=(
                     f"'{y_pred}' may not grow well in {soil_type} soil.\n\n"
                    f" Suggested crops for {soil_type} soil:\n{alternatives}"
                    ),
                    font=("Tahoma", 15)
                    )
                    
        else:
            messagebox.showerror("Missing Soil Type", "Please upload a soil image before predicting.")
            return
        
        result.grid(row=20, column=0, columnspan=50, pady=18)
        
    except ValueError:
        messagebox.showerror("Type Error", "Please enter valid numeric values only.")
 
b1=Button(root,text='Submit',command=submit_click,width=20, height=2)
b1.grid(row=17,column=0)

upload_btn = Button(root, text="Upload Soil Image", command=upload_and_classify_soil_image, width=20, height=2)
upload_btn.grid(row=16, column=0, pady=10)

def clear_entries():
    for i in range(0,7):
         entries[i].delete(0,END)
    result.config(text="")
    result.grid_forget()      
    global soil_type
    soil_type = None
    city_entry.delete(0,END)


clear_button = Button(root, text="Clear", command=clear_entries,width=20, height=2)
clear_button.grid(row=17,column=7)


# below part for simulation only 
sim_label = Label(root, text="", bg='#8A9A5B', fg="black", font=("Tahoma", 11), width=65, justify="left")
#sim_label.grid(row=18, column=0, columnspan=10) # uncomment when testing the sensor data

def listen_to_pi():
    global client_socket, listening
    buffer = ""
    try:
        while listening:
            data = client_socket.recv(1024).decode()
            if not data:
                break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                print("[DEBUG] Received line:", line)

                try:
                    temp_str, humid_str = line.strip().split(",")
                    temp = float(temp_str)
                    humid = float(humid_str)
                    print(f"temp: {temp} humid: {humid}")
                    root.after(0, handle_pi_data, temp, humid)
                except ValueError:
                    continue
    except Exception as e:
        sim_label.config(text=f"Socket error: {e}")
        stop_receiving_from_pi()
        
def handle_pi_data(temp, humid):
    try:
        # Get N, P, K, pH, Rainfall manually from GUI
        entries[3].delete(0, END)
        entries[3].insert(0, f"{temp:.2f}")
        entries[4].delete(0, END)
        entries[4].insert(0, f"{humid:.2f}")
        values = [e.get().strip() for e in entries]
        if any(v == '' for v in values):
            sim_label.config(text="Please fill N, P, K, pH, and Rainfall manually.")
            return

        N = float(values[0])
        P = float(values[1])
        K = float(values[2])
        ph = float(values[5])
        rain = float(values[6])

        # Update GUI entries for temp and humidity from Pi
        

        # Call your model
        prediction = getData(N, P, K, temp, humid, ph, rain)
        print(prediction)

        # Save and show
        file_exists = os.path.isfile("pi_live_data.csv")
        with open("pi_live_data.csv", "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "Recommended_crop"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "N": N, "P": P, "K": K,
                "temperature": temp,
                "humidity": humid,
                "ph": ph,
                "rainfall": rain,
                "Recommended_crop": prediction
            })

        sim_label.config(text=(f"Live Sensor Input:\n"
                               f"NPK = ({N}, {P}, {K}) | Temp = {temp}°C | Humidity = {humid}%\n"
                               f"pH = {ph} | Rainfall = {rain} mm\n"
                               f"Recommended Crop: {prediction}"))

    except Exception as e:
        sim_label.config(text=f"Error: {e}")


def start_receiving_from_pi():
    global client_socket, listener_thread, listening
    if not listening:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            listening = True
            listener_thread = threading.Thread(target=listen_to_pi, daemon=True)
            listener_thread.start()
            sim_label.config(text="Connected to Raspberry Pi. Receiving live data...")
        except Exception as e:
            messagebox.showerror("Connection Error", f"Could not connect to Raspberry Pi: {e}")

def stop_receiving_from_pi():
    global listening, client_socket
    listening = False
    if client_socket:
        try:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
        except:
            pass
    sim_label.config(text="Disconnected from Raspberry Pi.")

sim_btn = Button(root, text="Start receive", command=start_receiving_from_pi, width=20, height=2)
sim_btn.grid(row=19, column=0, pady=10)

stop_btn = Button(root, text="Stop receive", command=stop_receiving_from_pi, width=20, height=2)
stop_btn.grid(row=19, column=7, pady=10)

root.mainloop()