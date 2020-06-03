# Fraud-Detection

1. Bu çalışma, Veri Bilimi Okulu tarafından organize edilen Data Science Bootcamp eğitimini bitirme projesi olarak; Muhammed Çakmak, İsmail Kaya, Muhammed Cimci, Ümit Ceylan, Berkan Acar ve Mert Ozan İnal' ın içinde bulunduğu 6 kişilik bir ekip tarafından yapıldı. 

Çalışmanın verileri 2019 yılı september ve october ayları arasında Kaggle platformunda IEEE-CIS organizasyonu tarafından düzenlenen Fraud Detection yarışmasından alındı ve kurduğumuz modellerin başarısı yine bu yarışma çerçevesinde test edildi. 

Muhammed Çakmak, İsmail Kaya, Muhammed Cimci, Ümit Ceylan, Berkan Acar ve Mert Ozan İnal' ın katkılarıyla yaptığımız verisiyle ilgili çalışmalarımızın özetini aşağıda açıklamış olacağız. yarışmanın amacı neydi? 

Can you detect fraud from customer transactions?
In this competition you are predicting the probability 
that an online transaction is fraudulent, as denoted by the binary target isFraud.

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
The data is broken into two files identity and transaction, which are joined by TransactionID. 
Not all transactions have corresponding identity information.


# Muhammed Çakmak, İsmail Kaya, Mert Ozan İnal, Muhammed Cimci, Ümit Ceylan, Berkan Acar ile ortak bir çalışma. 

#Projeyi yarışma amaçlı katılmadık. bir eğitim kapsamında bitirme projesi olarak yaptık.

Final modeli oluşturmak için modeli eğitirken, eğitim ve test verilerinin toplam hacmi 5 GB’ ın üstündeydi.
Tabloları birleştirip veriyi manipüle etmeye başladığımızda eğitim verimizde; 590.540 gözlem – 434 değişken,
test verimizde ise 506.691 gözlem – 433 değişken vardı.

# •	Dosya hacmini küçülttük. sonucta ... kadar gözlem, ... değişken oldu ve dosya boyutu memory reduce yapılarak ... gb boyutuna indirildi.

•	Eksik verileri – uç değerleri – korelasyonları basit düzeyde incelemek.

•	Tüm bu işlemler sonunda korelasyonları inceleyerek 0.85-0.90 üstü korelasyonlu olan değişkenlerin 
birini veri setinden çıkarmak (regresyon problemi için çoklu doğrusal bağlantıyı engellemek).

# model kısnı

•	Hangi modeller kullanılacağını belirlemek.
•	En uygun model belirleyerek parametre ayarlamalarını yapmak, modeli geliştirmek.
•	Gerekli ise veri manipülasyonu ve tekrar modelin kurulması
- 3 model xgboost,lightgbm ve catboost modellerini kullanarak grid search yöntemi kullanarak hiperparametre tuning yaptık.
- her üç model için kfold kullanarak birer tahmin yaptık ayrıca modellerin sade halleriyle tahmin yaptık ve bunların başarısını karşılaştırdık.
- 6 modelin sonuçları tablo ile gösterilecek.
- catboost modelinde kategorik featurelara duyarlılığını test ettik. kategorik feature özelliği tercih edildiğinde sonucun her deneme yüksek başarı g
göstermesinden dolayı kategorik feature özelliği kullanıldı.
- başarı metrikleri denendi.


- encoding yöntemleri
- feature engineering yöntemleri
- değişken eleme, çıkarma, boyut düşürme, korelasyon vs. yöntemleri
- Featureları 6 kişiye paylaştırıp detaylı inceleme yaptık. feature lar ile ilgili hareket tarzı geliştirdik.
- Feature engineering yöntemleri geliştirdik.




