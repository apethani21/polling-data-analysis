import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def read_data(folder):
    data = []
    files = os.listdir(f"./{folder}/")
    for file in files:
        with open(f"./{folder}/{file}", "r") as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)
    return df


parties = ("GRN", "LAB", "LDEM", "SNP", "CON", "BREX", "REFUK", "UKIP", "OTH")
color_dict = {
    "GRN": "#77AB43",
    "LAB": "#FF2700",
    "LDEM": "#FAA61A",
    "SNP": "#EFF10A",
    "CON": "#008FD5",
    "BREX": "#12B6CF",
    "REFUK": "#12B6CF",
    "UKIP": "#B3009D",
    "OTH": "#000000",
}


def extract_party_results(result):
    process_results = {}
    for party_result in result:
        try:
            party, poll_perc, change = party_result.split()
            party = party.rstrip(":")
            # label change for Reform UK
            if party == "REF":
                party = "REFUK"
            assert party in parties, f"Unrecognised party {party}"
            poll_perc = int(
                poll_perc.strip()[:-1] if poll_perc.endswith("%") else poll_perc.strip()
            )
            change = change[1:] if change.startswith("(") else change
            change = change[:-1] if change.endswith(")") else change
            change = 0 if change in {"-", "–", "="} else int(change)
            process_results[party] = {"poll_perc": poll_perc, "change": change}
        except Exception as e:
            raise Exception(f"Error processing result {result})") from e
    return process_results


def _process_time_range(time_range):
    if pd.isnull(time_range):
        return np.nan

    time_range = time_range.strip()
    split_time_range = time_range.split()

    if re.match("^[0-9]{1,2} [A-Z][a-z]{2}$", time_range):
        return time_range, time_range

    if re.match("^[0-9]{1,2} - [0-9]{1,2} [A-Z][a-z]{2}$", time_range):
        month = split_time_range[-1].strip()
        start = f"{split_time_range[0]} {month}"
        end = f"{split_time_range[2]} {month}"
        return start, end

    if re.match("^[0-9]{1,2} [A-Z][a-z]{2} - [0-9]{1,2} [A-Z][a-z]{2}$", time_range):
        start_month = split_time_range[1].strip()
        end_month = split_time_range[-1].strip()
        start = f"{split_time_range[0]} {start_month}"
        end = f"{split_time_range[3]} {end_month}"
        return start, end


def _process_range_date(time_range, created_at, side, orig_row):
    """side is "start" or "end" """
    if side == "start":
        date_clean = time_range[0] if not pd.isnull(time_range) else time_range
    elif side == "end":
        date_clean = time_range[1] if not pd.isnull(time_range) else time_range
    else:
        raise ValueError(f"Invalid input for side: {side}, should be start or end.")
    if pd.isnull(date_clean):
        return date_clean

    else:
        try:
            # to deal with a typo in some tweets..
            if "Jus" in date_clean:
                date_clean = date_clean.replace("Jus", "Jun")
            if created_at.month >= pd.to_datetime(f"{date_clean} 1900").month:
                date_clean = pd.to_datetime(
                    f"{date_clean} {created_at.year}", dayfirst=True
                )
            else:
                date_clean = pd.to_datetime(
                    f"{date_clean} {created_at.year - 1}", dayfirst=True
                )
            return date_clean
        except Exception as e:
            raise Exception(f"Bad row: {orig_row}") from e


def process_wvi_data(df):
    df.rename(columns={"_id": "id"}, inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df.sort_values(by="created_at", inplace=True, ignore_index=True)
    poll_results_regex = "^([A-Z]){3,4}:[ ]{1,}[0-9]{1,3}%[ ]{1,}\((.)?([0-9]{1,3})?\)$"
    df["full_text"] = df["full_text"].str.replace("\xa0", "")
    df["full_text"] = df["full_text"].str.split("\n")

    df["results"] = (
        df["full_text"]
        .apply(lambda x: [text for text in x if re.match(poll_results_regex, text)])
        .apply(extract_party_results)
    )

    df["sources_info"] = (
        df["full_text"]
        .apply(lambda x: [text for text in x if "via " in text.lower()])
        .apply(lambda x: x[0] if len(x) > 0 else "")
        .str.split(",")
    )
    df["source"] = (
        df["sources_info"]
        .apply(lambda x: x[0][3:].strip() if len(x) > 0 else np.nan)
        .replace(
            {
                "@YouGov https://t.co/z0lyoLiIMv": "YouGov",
                "@RedfieldWIlton": "RedfieldWilton",
                "Redfield &amp; Wilton": "RedfieldWilton",
                "@OpiniumResearch https://t.co/bfyhsXXkrP": "OpiniumResearch",
            }
        )
        .apply(lambda x: x[1:] if (not pd.isnull(x)) and x.startswith("@") else x)
    )

    df["time_range"] = (
        df["sources_info"]
        .apply(lambda x: x[1].strip() if len(x) > 1 else np.nan)
        .apply(_process_time_range)
    )  # column will now contain (start, end)

    # df["start"] = df["time_range"].apply(lambda x: x[0] if not pd.isnull(x) else x)
    # df["start"] = pd.to_datetime(df["start"] + " " + df["created_at"].dt.year.astype(str), dayfirst=True)
    # df["end"] = df["time_range"].apply(lambda x: x[1] if not pd.isnull(x) else x)
    # df["end"] = pd.to_datetime(df["end"] + " " + df["created_at"].dt.year.astype(str), dayfirst=True)  # handle year crossover
    df["start"] = df.apply(
        lambda x: _process_range_date(x["time_range"], x["created_at"], "start", x),
        axis=1,
    )
    df["end"] = df.apply(
        lambda x: _process_range_date(x["time_range"], x["created_at"], "end", x),
        axis=1,
    )

    df["temp_index"] = df["end"].copy()
    df["temp_index"].fillna(df["created_at"], inplace=True)

    df["change_info"] = (
        df["full_text"]
        .apply(lambda x: [text for text in x if "chg" in text.lower()])
        .apply(lambda x: x[0] if x else np.nan)
        .str.lstrip("Chgs. w/ ")
    )

    for party in parties:
        df[party] = df["results"].apply(
            lambda x: x.get(party, {}).get("poll_perc", np.nan)
        )
        df[f"{party}_change"] = df["results"].apply(
            lambda x: x.get(party, {}).get("change", np.nan)
        )

    assert df["id"].is_unique

    df.set_index("temp_index", inplace=True, drop=True)
    df.drop(
        columns=["full_text", "results", "sources_info", "time_range", "id"],
        inplace=True,
    )
    df["collection_source"] = "britainelects"
    df.index.name = "date"
    return df


def plot_vote_intention(
    df,
    add_lockdown_context=False,
    add_lifetime_context=False,
    agg=None,
    markersize=None,
):
    """agg: D, W, M, Q etc."""
    if agg is not None:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(20, 15), gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    df[list(parties)].plot(
        ax=ax,
        linewidth=1 if agg is None else 0,
        marker="o",
        markersize=markersize if markersize is not None else (3 if agg is None else 2),
        alpha=1 if agg is None else 0.75,
        color=[color_dict[col] for col in df[list(parties)].columns],
        legend=True,
    )

    if agg is not None:
        (
            df.resample(agg, label="right")
            .mean()[list(parties)]
            .plot(
                linewidth=3,
                color=[color_dict[col] for col in df[list(parties)].columns],
                ax=ax,
                label=None,
                zorder=10,
            )
        )
        df.resample(agg, label="right").count()[list(parties)].plot(
            linewidth=1,
            color=[color_dict[col] for col in df[list(parties)].columns],
            ax=ax2,
        )
        ax2.legend(bbox_to_anchor=(1, 0.5), loc="center left", borderaxespad=0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1, 0.5),
        loc="center left",
        borderaxespad=0,
    )
    ax.set_xlim(
        (df.index.min() - pd.Timedelta(14, "D")).date(),
        (df.index.max() + pd.Timedelta(14, "D")).date(),
    )
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("% Poll", fontweight="bold")
    if agg is None:
        ax.set_title(f"Westminster Voting Intention", fontweight="bold", y=1.1)
    else:
        agg_fullname = {
            "D": "Daily",
            "W": "Weekly",
            "M": "Monthly",
            "SM": "Semi-Month",
            "Q": "Quarterly",
        }.get(agg, agg)
        ax.set_title(
            f"Westminster Voting Intention - {agg_fullname} Aggregation",
            fontweight="bold",
            y=1.1,
        )
        ax2.set_ylabel(f"Polls - {agg_fullname} counts")

    if add_lockdown_context:
        ax.axvline(
            pd.to_datetime("23 March 2020"), color="dimgrey", ymin=0.045, linewidth=3
        )
        ax.axvline(
            pd.to_datetime("10 May 2020"), color="dimgrey", ymin=0.045, linewidth=3
        )

        ax.axvspan(
            pd.to_datetime("23 March 2020"),
            pd.to_datetime("10 May 2020"),
            alpha=0.25,
            color="red",
            ymin=0.045,
        )
        ax.text(
            pd.to_datetime("17 Apr 2020"),
            16,
            "First \n lockdown",
            style="italic",
            fontweight="bold",
            horizontalalignment="center",
            fontsize=12,
        )

        ax.axvline(
            pd.to_datetime("4 April 2020"),
            color="dimgrey",
            ymin=0.045,
            linewidth=3,
            alpha=0.4,
        )
        ax.text(
            pd.to_datetime("12 Apr 2020"),
            35,
            "Kier Starmer \n becomes \n Labour leader \n 4 Apr",
            style="italic",
            horizontalalignment="center",
            fontsize=9,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("5 Nov 2020"), color="dimgrey", ymin=0.045, linewidth=3
        )
        ax.axvline(
            pd.to_datetime("2 Dec 2020"), color="dimgrey", ymin=0.045, linewidth=3
        )
        ax.axvspan(
            pd.to_datetime("5 Nov 2020"),
            pd.to_datetime("2 Dec 2020"),
            alpha=0.25,
            color="red",
            ymin=0.045,
        )
        ax.text(
            pd.to_datetime("18 Nov 2020"),
            16,
            "Second \n lockdown",
            style="italic",
            fontweight="bold",
            horizontalalignment="center",
            fontsize=8,
        )

        ax.axvline(
            pd.to_datetime("6 Jan 2021"), color="dimgrey", ymin=0.045, linewidth=3
        )
        ax.axvline(
            pd.to_datetime("29 Mar 2021"), color="dimgrey", ymin=0.045, linewidth=3
        )
        ax.axvspan(
            pd.to_datetime("6 Jan 2021"),
            pd.to_datetime("29 Mar 2021"),
            alpha=0.25,
            color="red",
            ymin=0.045,
        )
        ax.text(
            pd.to_datetime("15 Feb 2021"),
            16,
            "Third \n lockdown",
            style="italic",
            fontweight="bold",
            horizontalalignment="center",
            fontsize=11,
        )
    if add_lifetime_context:
        ax.axvline(
            pd.to_datetime("22 Jan 2013"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("22 Jan 2013"),
            54.5,
            "Cameron offers \n EU Ref upon \n winning next GE",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("8 May 2015"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("8 May 2015"),
            54.5,
            "Cameron \n remains PM \n maj. 12",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("23 June 2016"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("23 June 2016"),
            54.5,
            "EU Referendum. \n 3 weeks later, May \n becomes PM",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("9 June 2017"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("9 June 2017"),
            54.5,
            "May calls \n snap election, \n minority govt \n C&S with DUP",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("1 Jan 2019"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("15 Sept 2018"),
            54.5,
            "Commons rejects WA \n  and no deal \n (in principle).\n Commons backs \n Brady amendment",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("24 July 2019"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("24 July 2019"),
            54.5,
            "Johnson \n becomes PM \n maj. 80",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("23 March 2020"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("23 March 2020"),
            54.5,
            "COVID \n LD 1",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("5 Nov 2020"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("5 Dec 2020"),
            54.5,
            "COVID \n Lockdowns \n 2 & 3",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        ax.axvline(
            pd.to_datetime("6 Jan 2021"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )

        ax.axvline(
            pd.to_datetime("8 Dec 2021"),
            color="dimgrey",
            ymin=0.045,
            ymax=0.96,
            linewidth=3,
        )
        ax.text(
            pd.to_datetime("8 Dec 2021"),
            54.5,
            "Partygate \n begins",
            style="italic",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

    return fig, ax


def null_values_plot(df, date_col="date", freq="M"):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Null Values Analysis", weight="bold")
    fig.subplots_adjust(top=0.9)

    gs = GridSpec(
        2, 2, wspace=0.01, hspace=0.05, width_ratios=[2, 1], height_ratios=[3, 1]
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = df[date_col].copy()
    sns.heatmap(df.T.isnull(), cbar=False, ax=ax1, cmap="binary")
    null_counts = df.isnull().sum(axis="rows")
    null_counts.plot.barh(ax=ax2, color="#4e5a65", width=0.95, align="edge", zorder=10)
    for p in ax2.patches:
        ax2.annotate(
            p.get_width(),
            (p.get_x() + p.get_width(), p.get_y() + 0.1),
            xytext=(5, 10),
            textcoords="offset points",
        )
    df.resample(freq).size().plot(ax=ax3, color="#4e5a65", linewidth=2, zorder=10)

    ax1.set_xticks([])
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.spines["left"].set_visible(False)
    ax1.set_yticklabels(
        ax1.get_yticklabels(),
        rotation=0,
        horizontalalignment="right",
        verticalalignment="baseline",
    )

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Null values heatmap")
    ax2.set_title("Number of null values")
    ax3.set_title(f"Total rows. Frequency = {freq}")
