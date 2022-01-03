#%%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#Datayı oku
data = pd.read_excel("./Online_Retail.xlsx")
print(data.head())

#Datanın sütunları
print(data.columns)

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

def hot_enconde(x):
    if(x<=0):
        return 0
    if(x>= 1):
        return 1

def ModelCreateAndAnalyze(Transactions):
    frequent_items = apriori(Transactions, min_support= 0.05, use_colnames=True)
    #Çıkarsanan kuralları bir dataframe içinde toplama
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
    print(rules.head())
    
print("Kullanılabilinir ülkeler\n{}".format(data.Country.unique()))

isExit = "no"

while isExit != "e": 
    try:
        Country = input("Kullanılabilinir ülkelerden birini yazınız: ")
        Transactions = AddToBasket(Country)
        Transactions_Encoded = Transactions.applymap(hot_enconde)
        Transactions = Transactions_Encoded
        ModelCreateAndAnalyze(Transactions)
        isExit = input("Çıkmak için e yazınız devam etmek için herhangi bir şey yazın:")
    except MemoryError:
        print("Yetersiz Bellek Başka Bir Ülke Dene İstersen")
    except:
        print("Böyle bir ülke listede yok")

