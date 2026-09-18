import os
for root, dirs, files in os.walk("/kaggle/input"):
    print("DIR:", root, "->", dirs)
    for f in files[:5]:
        print("   FILE:", f)
