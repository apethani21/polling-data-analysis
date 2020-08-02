import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(folder):
    data = []
    files = os.listdir(f"./{folder}/")
    for file in files:
        with open(f"./{folder}/{file}", "r") as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)
    return df


parties = ("GRN", "LAB", "LDEM", "SNP", "CON", "BREX", "UKIP")
color_dict = {
    "GRN": "#77AB43",
    "LAB": "#FF2700",
    "LDEM": "#FAA61A",
    "SNP": "#EFF10A",
    "CON": "#008FD5",
    "BREX": "#12B6CF",
    "UKIP": "#B3009D"
}


def extract_party_results(result):
    process_results = {}
    for party_result in result:
        party, poll_perc, change = party_result.split()
        party = party.rstrip(":")
        assert party in parties
        poll_perc = int(poll_perc.strip()[:-1] if poll_perc.endswith("%") else poll_perc.strip())
        change = change[1:] if change.startswith("(") else change
        change = change[:-1] if change.endswith(")") else change
        change = 0 if change == '-' else int(change)
        process_results[party] = {
            "poll_perc": poll_perc,
            "change": change
        }
    return process_results


def process_time_range(time_range):
    if pd.isnull(time_range):
        return np.nan

    time_range = time_range.strip()
    split_time_range = time_range.split()
    
    if re.match('^[0-9]{1,2} [A-Z][a-z]{2}$', time_range):
        return time_range, time_range

    if re.match('^[0-9]{1,2} - [0-9]{1,2} [A-Z][a-z]{2}$', time_range):
        month = split_time_range[-1].strip()
        start = f"{split_time_range[0]} {month}"
        end = f"{split_time_range[2]} {month}"
        return start, end

    if re.match('^[0-9]{1,2} [A-Z][a-z]{2} - [0-9]{1,2} [A-Z][a-z]{2}$', time_range):
        start_month = split_time_range[1].strip()
        end_month = split_time_range[-1].strip()
        start = f"{split_time_range[0]} {start_month}"
        end = f"{split_time_range[3]} {end_month}"
        return start, end


def process_wvi_data(df):
    df.rename(columns={"_id": "id"}, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df.sort_values(by="created_at", inplace=True, ignore_index=True)
    poll_results_regex = "^([A-Z]){3,4}:[ ]{1,}[0-9]{1,3}%[ ]{1,}\((.)?([0-9]{1,3})?\)$"
    df['full_text'] = df['full_text'].str.split("\n")

    df["results"] = (df['full_text']
                     .apply(lambda x: [text for text in x if re.match(poll_results_regex, text)])
                     .apply(extract_party_results))

    df["sources_info"] = (df['full_text']
                          .apply(lambda x: [text for text in x if 'via ' in text.lower()][0])
                          .str.split(','))

    df["source"] = (df["sources_info"]
                    .apply(lambda x: x[0][3:].strip() if len(x) > 0 else np.nan)
                    .replace({
                        "@YouGov https://t.co/z0lyoLiIMv": "@YouGov",
                        "@RedfieldWIlton": "RedfieldWilton",
                        "Redfield &amp; Wilton": "RedfieldWilton"
                    })
                    .apply(lambda x: x[1:] if (not pd.isnull(x)) and x.startswith('@') else x))

    df["time_range"] = (df["sources_info"]
                        .apply(lambda x: x[1].strip() if len(x) > 1 else np.nan)
                        .apply(process_time_range))

    df["start"] = df["time_range"].apply(lambda x: x[0] if not pd.isnull(x) else x)
    df["end"] = df["time_range"].apply(lambda x: x[1] if not pd.isnull(x) else x)
    df["start"] = pd.to_datetime(df["start"] + " " + df["created_at"].dt.year.astype(str), dayfirst=True)
    df["end"] = pd.to_datetime(df["end"] + " " + df["created_at"].dt.year.astype(str), dayfirst=True)
    df["temp_index"] = df["end"].copy()
    df["temp_index"].fillna(df["created_at"], inplace=True)

    df["change_info"] = (df["full_text"]
                         .apply(lambda x: [text for text in x if 'chg' in text.lower()])
                         .apply(lambda x: x[0] if x else np.nan)
                         .str.lstrip("Chgs. w/ "))

    for party in parties:
        df[party] = df["results"].apply(lambda x: x.get(party, {}).get("poll_perc", np.nan))
        df[f"{party}_change"] = df["results"].apply(lambda x: x.get(party, {}).get("change", np.nan))

    assert df["id"].is_unique

    df.set_index('temp_index', inplace=True, drop=True)
    df.drop(columns=["full_text", "results", "sources_info", "time_range", "id"], inplace=True)
    df["collection_source"] = "britainelects"
    df.index.name = "date"
    return df


def plot_vote_intention(df, add_context_annot=False):
    df[list(parties)].plot(figsize=(20, 10),
                           linewidth=1,
                           marker='o',
                           markersize=5,
                           color=[color_dict[col] for col in df[list(parties)].columns])

    plt.xlim((df.index.min() - pd.Timedelta(14, 'D')).date(), (df.index.max() + pd.Timedelta(14, 'D')).date())
    plt.xlabel("Date", fontweight='bold')
    plt.ylabel("% Poll", fontweight='bold')
    plt.title("Westminster Voting Intention", fontweight='bold')
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", borderaxespad=0)

    if add_context_annot:
        plt.axvline(pd.to_datetime("23 March 2020"), color="black", ymin=0.045, linewidth=3)
        plt.axvline(pd.to_datetime("10 May 2020"), color="black", ymin=0.045, linewidth=3)

        plt.axvspan(pd.to_datetime("23 March 2020"), pd.to_datetime("10 May 2020"), alpha=0.25, color='red', ymin=0.045)
        plt.text(pd.to_datetime("17 Apr 2020"), 20, "Height of lockdown",
                 style='italic', fontweight='bold', horizontalalignment='center', fontsize=20)
        plt.text(pd.to_datetime("17 Apr 2020"), 17, "Stay at home → Stay Alert", style='italic', horizontalalignment='center')

        plt.axvline(pd.to_datetime("4 April 2020"), color="black", ymin=0.045, linewidth=3, alpha=0.4)
        plt.text(pd.to_datetime("12 Apr 2020"), 35, "Kier Starmer \n becomes \n Labour leader \n 4 Apr", 
                 style='italic', horizontalalignment='center', fontsize=10, fontweight='bold');
