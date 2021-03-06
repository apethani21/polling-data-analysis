# Analysing aggregated UK polling data

- The tweet collection & storage pipeline used in [email-service](https://github.com/apethani21/email-service) is used to also collect tweets from [@britainelects](https://twitter.com/britainelects).
- `extract_britain_elects.py` queries the necessary data from my MongoDB database and saves it. Projections allow for querying specific fields within each document, keeping only the tweet body and timestamp in this case.
- The additional polling historical data in uk_polling_report_historical.csv was scraped from [UK Polling Report](http://ukpollingreport.co.uk/voting-intention-2).
- `polling_report_history.py` contains a function to clean this data to combine with the Westminster Voting Intention Twitter data from Britain Elects.
- `analyse_polling_data.ipynb` contains various bits of analysis using the combined data, as well as a quick look at the sort of text found in general within all of Britain Elects' tweets, including those unrelated to Westminster Voting Intention. This data can be found in the [britain_elects_all](https://github.com/apethani21/polling-data-analysis/tree/master/britain_elects_all) folder.

![Monthly WVI](https://github.com/apethani21/polling-data-analysis/blob/master/wvi_monthly_average.png)
