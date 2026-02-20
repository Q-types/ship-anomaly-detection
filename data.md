python - << 'EOF'
import pandas as pd

url = "https://raw.githubusercontent.com/fourthrevlxd/cam_dsb/main/engine.csv"
df = pd.read_csv(url)

df.to_csv("data/train.csv", index=False)
print("Saved data/train.csv")
EOF