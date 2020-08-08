import pandas as pd

def read_historical_polling_report():
    histo = pd.read_csv("uk_polling_report_historical.csv")
    histo["end"] = pd.to_datetime(histo["end"], dayfirst=False, yearfirst=True)
    histo["source"] = (histo["source"].str.split("/")
                       .apply(lambda x: x[0])
                       .str.replace(" ", "")
                       .str.replace("-", "")
                       .replace({
                           "RedfieldandWilton": "RedfieldWilton",
                           "Opinium": "OpiniumResearch",
                           "BMG": "BMGResearch",
                           "Deltapoll": "DeltapollUK",
                           "Kantar": "KantarPublic"
                       }))
    histo.drop(columns='CON_LEAD', inplace=True)
    histo.drop_duplicates(inplace=True, ignore_index=True)
    histo = histo[histo["end"] < pd.to_datetime("2020-01-10", dayfirst=False)]
    histo.set_index("end", drop=False, inplace=True)
    histo.sort_index(ascending=False)
    histo.index.name = "date"
    histo["collection_source"] = "uk_prh"
    return histo
