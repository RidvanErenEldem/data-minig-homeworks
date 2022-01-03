#%%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#Datayı oku
data = pd.read_excel("./Online_Retail.xlsx")
print(data.head())

#Datanın sütunları
print(data.columns)
#Datadaki ülkeler
print(data.Country.unique())

#Data üzerinde temizlik işlmeleri
# Description(Ürün Adı) kısmındaki boşluklar fazladan boşluklar çıkartılıyor
data['Description'] = data['Description'].str.strip()

# fatura no olmayan veriler siliniyor
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# kredi ile yapılınan bütün işlemler siliniyor
data = data[~data['InvoiceNo'].str.contains('C')]


#Verileri işlem bölgesine göre sınıflandırma

def AddToBasket(CountryName):
    basket = (data[data['Country'] == CountryName]
                       .groupby(['InvoiceNo', 'Description'])['Quantity']
                       .sum().unstack().reset_index().fillna(0)
                       .set_index('InvoiceNo'))
    return basket

Transactions_France = AddToBasket("France")
Transactions_UK = AddToBasket("United Kingdom")
Transactions_Portugal = AddToBasket("Portugal")
Transactions_Sweden = AddToBasket("Sweden")

# Verileri ilgili kütüphanelere uygun hale getirmek için hot_enconde fonksiyonunu tanımlama

def hot_enconde(x):
    if(x<=0):
        return 0
    if(x>= 1):
        return 1
    
#datasetleri encode etme

Transactions_Encoded = Transactions_France.applymap(hot_enconde)
Transactions_France = Transactions_Encoded

Transactions_Encoded = Transactions_UK.applymap(hot_enconde)
Transactions_UK = Transactions_Encoded

Transactions_Encoded = Transactions_Portugal.applymap(hot_enconde)
Transactions_Portugal = Transactions_Encoded

Transactions_Encoded = Transactions_Sweden.applymap(hot_enconde)
Transactions_Sweden = Transactions_Encoded

#Model oluşturma ve analiz etme 
def ModelCreateAndAnalyze(Transactions):
    frequent_items = apriori(Transactions, min_support= 0.05, use_colnames=True)
    #Çıkarsanan kuralları bir dataframe içinde toplama
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
    print(rules.head())

print("France")
ModelCreateAndAnalyze(Transactions_France)
print("UK")
ModelCreateAndAnalyze(Transactions_UK)
print("Portugal")
ModelCreateAndAnalyze(Transactions_Portugal)
print("Sweden")
ModelCreateAndAnalyze(Transactions_Sweden)
