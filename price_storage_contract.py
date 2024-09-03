
import pandas as pd

def price_storage_contract(
    nat_gas_data,
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_month,
    injection_withdrawal_cost,
):
    """
    Prices a commodity storage contract based on the provided parameters.

    Parameters:
    - nat_gas_data: DataFrame with columns 'Dates' and 'Prices' for the historical price data.
    - injection_dates: List of dates when the commodity will be injected into storage.
    - withdrawal_dates: List of dates when the commodity will be withdrawn from storage.
    - injection_rate: Rate at which gas can be injected (volume per month).
    - withdrawal_rate: Rate at which gas can be withdrawn (volume per month).
    - max_volume: Maximum volume that can be stored.
    - storage_cost_per_month: Monthly cost of storing the gas.
    - injection_withdrawal_cost: Cost per unit for injecting/withdrawing gas.

    Returns:
    - The value of the contract.
    """
    
    # Convert dates to datetime
    nat_gas_data['Dates'] = pd.to_datetime(nat_gas_data['Dates'], format='%m/%d/%y')
    nat_gas_data.set_index('Dates', inplace=True)
    
    # Calculate injection costs and volumes
    total_injected_volume = 0
    injection_cost = 0
    for date in injection_dates:
        price = nat_gas_data.loc[date]['Prices']
        volume = min(injection_rate, max_volume - total_injected_volume)
        injection_cost += price * volume + injection_withdrawal_cost
        total_injected_volume += volume
        max_volume -= volume
    
    # Calculate withdrawal revenues and volumes
    total_withdrawn_volume = 0
    withdrawal_revenue = 0
    for date in withdrawal_dates:
        price = nat_gas_data.loc[date]['Prices']
        volume = min(withdrawal_rate, total_injected_volume - total_withdrawn_volume)
        withdrawal_revenue += price * volume - injection_withdrawal_cost
        total_withdrawn_volume += volume
    
    # Calculate storage costs
    storage_duration = (pd.to_datetime(withdrawal_dates[-1]) - pd.to_datetime(injection_dates[0])).days // 30
    total_storage_cost = storage_cost_per_month * storage_duration
    
    # Calculate the final contract value
    contract_value = withdrawal_revenue - injection_cost - total_storage_cost
    
    return contract_value

# Sample test
file_path_new = 'Nat_Gas.csv'
nat_gas_data_new = pd.read_csv(file_path_new)

injection_dates = ['2023-06-30', '2023-07-31']
withdrawal_dates = ['2023-12-31', '2024-1-31']
injection_rate = 500000  # 500,000 MMBtu per month
withdrawal_rate = 500000  # 500,000 MMBtu per month
max_volume = 1000000  # 1,000,000 MMBtu maximum storage
storage_cost_per_month = 100000  # $100,000 per month
injection_withdrawal_cost = 10000  # $10,000 per injection/withdrawal

# Calculate the contract value
contract_value = price_storage_contract(
    nat_gas_data_new,
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_month,
    injection_withdrawal_cost
)

print(f"The value of the storage contract is: ${contract_value}")
