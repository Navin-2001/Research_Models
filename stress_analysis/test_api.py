import urllib.request
import json

url = "http://localhost:5001/analyze/patterns"
data = {
  "patterns": [
    {
      "duration_minutes": 150, 
      "frequency": 40, 
      "late_night_logins": 2
    },
    {
      "duration_minutes": 200, 
      "frequency": 55, 
      "late_night_logins": 5
    }
  ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'))
req.add_header('Content-Type', 'application/json')

try:
    with urllib.request.urlopen(req) as response:
        print("SUCCESS! The ML Model successfully received the data and returned this analysis:")
        print(response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print(f"FAILED with Status Code {e.code}:")
    print(e.read().decode('utf-8'))
