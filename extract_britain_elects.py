import os
import re
import json
import pandas as pd
from pymongo import MongoClient

client = MongoClient()
collection = client.twitter.britainelects

wvi_doc_filter = {
   "full_text": {
      "$regex": re.compile("Westminster voting intention", re.IGNORECASE)
   }
}

pm_preference_doc_filter = {
   "full_text": {
      "$regex": re.compile("Preference for Prime Minister", re.IGNORECASE)
   }
}
projection = ["created_at", "full_text"]

westminster_voting_docs = collection.find(filter=wvi_doc_filter, projection=projection)
os.makedirs("./britain_elects_wvi_data", exist_ok=True)
for doc in westminster_voting_docs:
    with open(f"./britain_elects_wvi_data/{doc['_id']}.json", "w") as f:
        json.dump(doc, f, indent=4)

pm_preference_docs = collection.find(filter=pm_preference_doc_filter, projection=projection)
os.makedirs("./britain_elects_pm_preference_data", exist_ok=True)
for doc in pm_preference_docs:
    with open(f"./britain_elects_pm_preference_data/{doc['_id']}.json", "w") as f:
        json.dump(doc, f, indent=4)
