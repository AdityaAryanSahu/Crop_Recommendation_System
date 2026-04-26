from datetime import datetime
from zoneinfo import ZoneInfo

# Get current time in IST
date = datetime.now(ZoneInfo('Asia/Kolkata'))

print(date)