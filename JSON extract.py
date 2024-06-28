import json
import pandas as pd

# import translation from ticker --> cik
ID_file_path = ('company_tickers_exchange.json')        # translate from ticker-->cik file
with open(ID_file_path, 'r') as id_file:
    data_ids = json.load(id_file)
df_ids = pd.DataFrame(data_ids['data'], columns=data_ids['fields'])
df_ids.set_index('ticker', inplace=True)


# create functions to easily switch between ticker & cik
def ticker_to_cik(tckr_str):
    # the SEC listing uses e.g. BRK-A instead of BRK.A
    return df_ids.loc[tckr_str.replace('.', '-')].cik


def cik_to_ticker(cik_int):
    return df_ids.index[df_ids['cik']==cik_int][0]


# read file with relevant tickers
ticker_filepath = "TICKERS/SP100_TICKER.txt"
with open(ticker_filepath, 'r') as ticker_file:
    tickers = ticker_file.read()
# open as list (split by linebreak) and strip leading/trailing spaces
tickers = tickers.split('\n')
tickers = [ticker.strip() for ticker in tickers]
# convert to CIK IDs
cik_ids = [ticker_to_cik(ticker) for ticker in tickers]
# make unique
cik_ids = set(cik_ids)


# prepare scanning SEC files
base_path = "file_path_to_JSONs"
# Prepare the relevant filenames in the SEC-JSON directory
cik_json_filenames = [f"CIK{str(id).zfill(10)}.json" for id in cik_ids]

# create empty list of dfs to populate later
df_list = []
for cik_json_filename in cik_json_filenames:
    print(cik_json_filename)
    # Open the JSON file for reading
    with open(base_path+cik_json_filename, 'r') as current_file:
        # Parse the JSON data into a Python dictionary
        current_data = json.load(current_file)

    try:
        company_cik = current_data['cik']
        call_dates_df = pd.DataFrame(current_data['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares'])
        call_dates_df['cik'] = company_cik
        call_dates_df['ticker'] = cik_to_ticker(company_cik)
        call_dates_df.set_index(['cik', 'filed'], inplace=True)
        df_list.append(call_dates_df)
    except:
        print(f"problem with file: {cik_json_filename}")

df = pd.concat(df_list)
print(df)


# print first available date of tickers
current_cik = 0
for cik, date in df.index:
    if current_cik != cik:
        ticker = df.loc[cik, date].ticker.iloc[0]
        print(f"CIK: {cik}, Ticker: {ticker} ----> {date}")
        current_cik = cik
