# check_path.py
import os

# The exact path your training script is looking for
file_path = "data/AP_NAQUIM_style_water_well_dataset_1000.csv"

print("-----------------------------------------")
print(f"👀 Checking for file at: {file_path}")

if os.path.exists(file_path):
    print("✅ SUCCESS: File was found!")
    print("You can now run the train_models.py script.")
else:
    print("❌ FAILED: File was NOT found at that path.")
    print("Please make sure you have a 'data' folder and the CSV file is inside it.")
print("-----------------------------------------")