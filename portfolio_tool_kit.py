import pandas as pd

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns
    Adjust the returns to normal values
    Convert the index to timeseries
    """
    returns = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv',
                       header=0, index_col=0, na_values=-99.99)
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    return returns

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def drawdown(return_series:pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
        the wealth index, 
        the previous peaks, and 
        the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def skewness(r):
    """
    Computes skewness of supplied series or a dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Computes kurtosis of supplied series or a dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4