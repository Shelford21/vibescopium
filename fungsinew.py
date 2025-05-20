import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re
import sys
import string
import csv
import requests
from io import StringIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from google_play_scraper import app, reviews_all, Sort, search , reviews
import nltk
from collections import Counter
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from wordcloud import WordCloud
import streamlit as st

if 'app_options' not in st.session_state:
    st.session_state['app_options'] = {}
if 'app_id' not in st.session_state:
    st.session_state['app_id'] = None
if 'reviews' not in st.session_state:
    st.session_state['reviews'] = None
if 'reset' not in st.session_state:
    st.session_state['reset'] = None
if 'csv' not in st.session_state:
    st.session_state['csv'] = None
if 'clean_df' not in st.session_state:
    st.session_state['clean_df'] = None
if 'positive_tweets' not in st.session_state:
    st.session_state['positive_tweets'] = None
if 'word_listpositive' not in st.session_state:
    st.session_state['word_listpositive'] = None
if 'word_listnegative' not in st.session_state:
    st.session_state['word_listnegative'] = None
if 'word_listnegative' not in st.session_state:
    st.session_state['word_listnegative'] = None
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state['tfidf_vectorizer'] = None
if 'do_stemming_choice' not in st.session_state:
    st.session_state['do_stemming_choice'] = None
if 'best_lr' not in st.session_state:
    st.session_state['best_lr'] = None
if 'eval_df' not in st.session_state:
    st.session_state['eval_df'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'y_pred_test_lr' not in st.session_state:
    st.session_state['y_pred_test_lr'] = None
if 'le' not in st.session_state:
    st.session_state['le'] = None

    
st.set_page_config(page_title="Vibe Scopium",
                   page_icon="🎭",
                   layout="wide")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS

load_css()



page_bg_img = """
<style>


</style>
"""

def cleaning_text(text):
            text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
            text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
            text = re.sub(r'RT[\s]', '', text)  # Remove RT
            text = re.sub(r"http\S+", '', text)  # Remove links
            text = re.sub(r'[0-9]+', '', text)  # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
            text = text.replace('\n', ' ')  # Replace new line with space
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations
            text = text.strip()  # Remove leading and trailing spaces
            return text
        
def case_folding_text(text):
                    return text.lower()
        
def tokenizing_text(text):
                    return word_tokenize(text)
        
# def filtering_text(text):
#                     list_stopwords = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
#                     custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
#                     list_stopwords.update(custom_stopwords)
#                     return [word for word in text if word not in list_stopwords]


# Create stopword list from Sastrawi
factoryz = StopWordRemoverFactory()
sastrawi_stopwords = set(factoryz.get_stop_words())

# Add your custom stopwords
custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
stopword_set = sastrawi_stopwords.union(custom_stopwords)

# Filtering function
def filtering_text(text):
    return [word for word in text if word not in stopword_set]

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# def stemming_text(text_list):
#             sentence = ' '.join(text_list)
#             return stemmer.stem(sentence).split()

# stem_cache = {}
# def stemming_text(text_list):
#     return [stem_cache.setdefault(word, stemmer.stem(word)) for word in text_list]

def stemming_text(text_list):
    return [stemmer.stem(word) for word in text_list]

        
            # def stemming_text(text_list):
            #             # Apply stemming on each word in the list
            #             return [stemmer.stem(word) for word in text_list]
        
            # def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
            # # Membuat objek stemmer
            #             factory = StemmerFactory()
            #             stemmer = factory.create_stemmer()
                    
            #             # Memecah teks menjadi daftar kata
            #             words = text.split()
                    
            #             # Menerapkan stemming pada setiap kata dalam daftar
            #             stemmed_words = [stemmer.stem(word) for word in words]
                    
            #             # Menggabungkan kata-kata yang telah distem
            #             stemmed_text = ' '.join(stemmed_words)
                    
            #             return stemmed_text
        
def to_sentence(list_words):
                    return ' '.join(list_words)
        
slang_words = {"@": "di", "ng":"menggunakan","nge":"menggunakan","abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "nnti": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}
        
def fix_slang_words(text):
                    return ' '.join(slang_words.get(word.lower(), word) for word in text.split())
                    
                    
        
                        # Check if text preprocessing has already been applied
                   
        
                    # Function to load and process data (cached)
@st.cache_data
def load_and_process_data(df):
    clean_df = df.copy(deep=True)  # Ensure deep copy

    clean_df['text_clean'] = clean_df['content'].apply(cleaning_text)
    clean_df['text_casefolding'] = clean_df['text_clean'].apply(case_folding_text)
    clean_df['text_slang_fixed'] = clean_df['text_casefolding'].apply(fix_slang_words)
    clean_df['text_tokenized'] = clean_df['text_slang_fixed'].apply(tokenizing_text)
    clean_df['text_stopword'] = clean_df['text_tokenized'].apply(filtering_text)

    # Check if stemming is enabled
    if st.session_state["do_stemming_choice"]: 
        clean_df['text_stemming'] = clean_df['text_stopword'].apply(stemming_text)
        clean_df['text_akhir'] = clean_df['text_stemming'].apply(to_sentence)
    else:
        clean_df['text_akhir'] = clean_df['text_stopword'].apply(to_sentence)

    return clean_df


# @st.cache_data
# def load_and_process_data(df):
#                         clean_df = df.copy(deep=True)  # Ensure deep copy
#                         clean_df['text_clean'] = clean_df['content'].apply(cleaning_text)
#                         clean_df['text_casefolding'] = clean_df['text_clean'].apply(case_folding_text)
#                         clean_df['text_slang_fixed'] = clean_df['text_casefolding'].apply(fix_slang_words)
#                         clean_df['text_tokenized'] = clean_df['text_slang_fixed'].apply(tokenizing_text)
#                         clean_df['text_stopword'] = clean_df['text_tokenized'].apply(filtering_text)
#                         #clean_df['text_stopwords'] = clean_df['text_stopword'].apply(to_sentence)
#                         #clean_df['text_stemming'] = clean_df['text_stopword'].apply(stemming_text)
#                         clean_df['text_stemming'] = clean_df['text_stopword'].apply(stemming_text) #klau stem pake ini
#                         #clean_df['text_akhir'] = clean_df['text_stopword'].apply(to_sentence) #klau ngga stem
#                         clean_df['text_akhir'] = clean_df['text_stemming'].apply(to_sentence) #dan ini stem
#                         return clean_df  # Return processed DataFrame


@st.cache_data
def fetch_lexicon(url):
        response = requests.get(url)
        if response.status_code == 200:
            reader = csv.reader(StringIO(response.text), delimiter=',')
            return {row[0]: int(row[1]) for row in reader}
        else:
            st.error(f"Failed to fetch lexicon data from {url}")
            return {}

@st.cache_data
def sentiment_analysis_lexicon_indonesia(text_list):
            results = []
            for text in text_list:
                score = sum(lexicon_positive.get(word, 0) for word in text)
                score += sum(lexicon_negative.get(word, 0) for word in text)
                polarity = 'positive' if score >= 0 else 'negative'
                results.append((score, polarity))
            return list(zip(*results))  # Returns tuple: (scores_list, polarity_list)

#st.markdown(page_bg_img, unsafe_allow_html=True)
def switch_page(page):
    st.session_state["current_page"] = page
    st.session_state["show_tweets_options"] = False
    st.experimental_rerun()
    

def check_reviews_threshold(num_reviews):
    if num_reviews is None:
        return "🚨 There are no reviews. Analysis could not be performed."
    elif num_reviews == 0:
        return "🚨 There are no reviews. Analysis could not be performed."
    elif num_reviews < THRESHOLD_LOW:
        return f"⚠️ Only {num_reviews} reviews found. Analysis may be less accurate."
    elif num_reviews < THRESHOLD_MEDIUM:
        return f"⚠️ {num_reviews} reviews found. Analysis can be performed, but results may be limited."
    else:
        return f"🎯 {num_reviews} reviews found. Perform sentiment analysis with enough data! "
    
    
# Adaptive thresholds for meaningful sentiment analysis
# Adaptive thresholds for meaningful sentiment analysis
global tfidf_vectorizer 
global best_lr
global reset
THRESHOLD_EMPTY = 1
THRESHOLD_LOW = 500
THRESHOLD_MEDIUM = 5001
import streamlit as st

import streamlit as st

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Vibe Scopium"

# # Function to navigate to "Vibe Scopium"
# def go_to_vibe_scopium():
#     st.session_state["current_page"] = "Vibe Scopium"
#     st.rerun()

# # Welcome Page
# if st.session_state["current_page"] == "Input App ID":
    
#     col1, col2, col3, col4, col5 = st.columns(5)
    
#     with col1:  # Center the button in col3
#         if st.button("Hi! , I would like to use the app", use_container_width=True):
#             go_to_vibe_scopium()
            
        

# # Vibe Scopium Page (after button is clicked)
# elif st.session_state["current_page"] == "Vibe Scopium":
#     st.sidebar.button("Vibe Scopium", disabled=True)  # Highlight active page


            
            # Page Buttons
            
if st.sidebar.button("🎭 Vibe Scopium"):
    switch_page("Vibe Scopium")
    
    
if st.session_state["current_page"] == "Vibe Scopium":
    # Create the container
    st.markdown(
        """
        <div class="transparent-container">
            <h1>🎭 VibeScopium</h1>
            <h2>Exploring Sentiments, Unveiling Insights</h2>
            <h4>
    Welcome to VibeScopium, your go-to sentiment analysis platform for Play Store apps! 📱✨<br><br>💡 Note: VibeScopium focuses exclusively on analyzing reviews written in Bahasa Indonesia, ensuring precise sentiment insights tailored to the Indonesian market. 🇮🇩<br><br>Whether you're a developer tracking user feedback, a marketer analyzing trends, or just curious about an app’s reputation, VibeScopium helps you dive deep into Play Store reviews. Our interactive dashboard scans and processes user feedback, revealing sentiment trends—positive, or negative—so you can understand what users truly think.<br><br>Simply enter an app name, explore real-time insights, and uncover valuable data through sentiment scores, keyword trends, and visual analytics 🔍📊.<br><br>So, get started, analyze app sentiments, and make informed decisions with VibeScopium. Let the data guide you! 🚀
</h4>
    
        """,
        unsafe_allow_html=True
    )

if st.sidebar.button("❓ How to use"):
    switch_page("How to use")
    
    
if st.session_state["current_page"] == "How to use":
    st.markdown(
        """
        <div class="transparent-container">
        <h3>❓ How to Use</h3>
        <h5> _

        
1.🔍 Navigate to the "Input App ID" section -
Go to the section labeled Input App ID on the sidebar or main screen.\n
2.⌨️ Type the name of the app you want to analyze -
In the input field, enter the name of the app whose reviews you’d like to scrape.\n
3.🟦 Click the "Find App" button -
Press the Find App button to search for apps matching your input.\n
4.📱 Select an app from the search results -
Choose one of the 5 apps listed based on your search.\n
5.⚙️ Decide whether to apply stemming
Choose Yes or No in the stemming option.
Yes will apply a linguistic normalization process (slower, more accurate),
No will skip it and process faster.\n
6.🔢 Specify the number of reviews to scrape -
Select how many user reviews you’d like to collect (e.g., 100, 500, 1000).\n
7.📥 Click the "Fetch Reviews" button -To start scraping the reviews.\n
8.✅ Done! Your data has been scraped successfully -
You can now visit other sections like DataFrames or Evaluation to explore visualizations and sentiment results.
</h5>
        </div>
    
        """,
        unsafe_allow_html=True
    )

if st.sidebar.button("🔍 Input App ID"):
    switch_page("Input App ID")

if st.session_state["current_page"] == "Input App ID":  
    st.markdown(
        """
        <div class="transparent-container">
            <h3>🔍 Start Scraping!</h3>
            <div class="transparent-container">
            <h5>
⚠️ Reviews that are below 5000 are considered low on data , therefore the prediction may be less accurate</h4>

    
        """,
        unsafe_allow_html=True
    )
    query = st.text_input("")

    # Create two columns
    col1, col2 ,a,s,d= st.columns(5)

    with col1:
         if st.button("Find App"):
            if not query or query.strip() == "":
                st.warning("Silakan isi nama aplikasi terlebih dahulu.")
            else:
                try:
                    results = search(query, lang='id', country='id')

                    if results:
                        st.session_state['app_options'] = {app['title']: app['appId'] for app in results[:5]}
                        st.session_state['app_id'] = None  # Reset app_id when new search is done
                        st.session_state['reviews'] = None  # Reset reviews when new search is done
                    else:
                        st.session_state['app_options'] = {}
                        st.session_state['app_id'] = None
                        st.write("Application not found.")
                except TypeError:
                    st.session_state['app_options'] = {}
                    st.session_state['app_id'] = None
                    st.write("Application not found.")

    with col2:
        if st.button("Reset Data"):
            st.session_state['reset'] = None
            st.session_state['app_id'] = None
            st.session_state['reviews'] = None
            st.session_state['app_reviews_df'] = None
            st.session_state['clean_df'] = None
            st.write("Data has been reset.")

    # Show selectbox only if there are app options
    if st.session_state['app_options']:
        selected_app = st.selectbox(
            "Choose Apps:", 
            list(st.session_state['app_options'].keys()), 
            key="selected_app"
        )

        # Update session state app_id when selection changes
        if selected_app:
            st.session_state['app_id'] = st.session_state['app_options'][selected_app]
            
          # NEW: Stemming option selectbox
    
        # Show the selectbox
        stemming_choice = st.selectbox(
            "Do you want to apply stemming to the text?",
            options=["No", "Yes"],
            index=0,
        )
        
        # Set session state accordingly
        if stemming_choice == "Yes":
            st.session_state["do_stemming_choice"] = True
            st.warning("⚠️ Enabling stemming may significantly increase processing time.")
        else:
            st.session_state["do_stemming_choice"] = None

    
    if st.session_state['app_id'] :
        app_id = st.session_state['app_id']
        st.write(f"App ID: {app_id}")
        # User input for the number of reviews to scrape
        count = st.slider("Number of reviews to fetch:", min_value=0, max_value=500000, step=500, value=10000)
        if st.button("Fetch reviews") :
            with st.spinner("Fetching reviews... (If there are many reviews, then scraping will take 1-5 minutes)"):
                reviews,_ = reviews(
                    app_id,
                    lang='id',
                    country='id',
                    sort=Sort.NEWEST,
                    count=count
                )
                if st.session_state['reset'] == True:
                    st.warning("Please press the 'Reset Data' button first")
                    st.session_state['reviews'] = None
                else:
                    st.session_state['reviews'] = reviews# Save reviews in session state
                    st.session_state['reset'] = True 
        
                    if not reviews:
                        st.write("🚨 No reviews found. Cannot proceed with analysis.")
                #st.success(f"Berhasil mengambil {len(reviews)} ulasan!")

        
        # Jika ulasan sudah ada di session state, tampilkan info & tombol download
        if 'reviews' in st.session_state and st.session_state['reviews']:
            reviews = st.session_state['reviews']
            # if not reviews:
            #     num_reviews = None
            # else:
            #     num_reviews = len(reviews)
            num_reviews = len(reviews)
        
            if num_reviews > 0:
                    # Simpan ke dalam buffer (tanpa menyimpan ke disk)
                    @st.cache_data  # Cache the CSV to avoid rerun issues
                    def generate_csv(reviews):
                        output = io.StringIO()
                        csv_writer = csv.writer(output)
                        csv_writer.writerow(['Review'])
                        for review in reviews:
                            csv_writer.writerow([review['content']])
                        return output.getvalue().encode('utf-8')
                
                    csv_bytes = generate_csv(reviews)
            
            if 'csv' not in st.session_state or st.session_state['csv'] is None:
                    st.session_state['csv'] = generate_csv(reviews) 
            
            st.write(check_reviews_threshold(num_reviews))

        reviews = st.session_state['reviews']        
        app_reviews_df = pd.DataFrame(reviews)
        st.session_state["app_reviews_df"] = app_reviews_df
        
                    # st.write("Dataset Shape:", app_reviews_df.shape)
                    # st.write("Sample Data:")
                    # st.write(app_reviews_df.head())
        
                    #app_reviews_df.to_csv('ulasan_aplikasi.csv', index=False)
        
                    # Fill missing values
        try:
            placeholder_date = pd.to_datetime("1900-01-01")
            app_reviews_df['repliedAt'] = app_reviews_df['repliedAt'].fillna(placeholder_date)
            app_reviews_df['replyContent'] = app_reviews_df['replyContent'].fillna("No reply")
            app_reviews_df['reviewCreatedVersion'] = app_reviews_df['reviewCreatedVersion'].fillna("1.1")
            app_reviews_df['appVersion'] = app_reviews_df['appVersion'].fillna("1.1")
        except Exception:
            st.write("")
                    # Clean data
        clean_df = app_reviews_df.dropna().drop_duplicates()
        st.session_state["clean_df"] = clean_df  # Store cleaned dataset in session state

         # Ensure clean_df is loaded before using it
        if "clean_df" not in st.session_state:
            st.error("Clean dataset is missing. Please load and clean data first.")
        else:
            try:# Load and preprocess data using cache
                clean_df = load_and_process_data(st.session_state["clean_df"])
                st.session_state["clean_df"] = clean_df
                #st.dataframe(clean_df)
            except Exception:
                 st.write("")


         # Load lexicons only once
        if "lexicon_positive" not in st.session_state:
            st.session_state["lexicon_positive"] = fetch_lexicon('https://raw.githubusercontent.com/Shelford21/vibescopium/main/lexicon_positive.csv')
    
        if "lexicon_negative" not in st.session_state:
            st.session_state["lexicon_negative"] = fetch_lexicon('https://raw.githubusercontent.com/Shelford21/vibescopium/main/lexicon_negative.csv')
    
        lexicon_positive = st.session_state["lexicon_positive"]
        lexicon_negative = st.session_state["lexicon_negative"]

        try:
            if "clean_df" in st.session_state:
                clean_df = st.session_state["clean_df"].copy()
        
                if "text_stopword" in clean_df.columns:
                    if "polarity_score" not in clean_df.columns or "polarity" not in clean_df.columns:
                        scores, polarities = sentiment_analysis_lexicon_indonesia(clean_df['text_stopword'])
                        clean_df['polarity_score'] = scores
                        clean_df['polarity'] = polarities
                        st.session_state["clean_df"] = clean_df
        
                    #st.write(clean_df[['content', 'text_stopword', 'polarity_score', 'polarity']].head())  
                else:
                    st.write("")
                    #st.dataframe(clean_df)
                    #st.error("Column 'text_stopword' is missing. Ensure text preprocessing is completed first.")
            #st.dataframe(clean_df,use_container_width=True , height=6000) 
        except Exception:
            st.write("_")

        #modelling and evalu\
        try:
            clean_df= st.session_state["clean_df"]
            X = clean_df['text_akhir']
            y = clean_df['polarity']
            data_size = len(clean_df)
    
            #tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, ngram_range=(1,2))

            @st.cache_resource
            def preprocess_tfidf(df):
                if "tfidf_vectorizer" not in st.session_state or st.session_state.tfidf_vectorizer is None:
                    st.session_state.tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1,2))  # Initialize
                    X_tfidf = st.session_state.tfidf_vectorizer.fit_transform(df['text_akhir'])
                else:
                    X_tfidf = st.session_state.tfidf_vectorizer.transform(df['text_akhir'])
                    #X_tfidf = tfidf_vectorizer.fit_transform(df['text_akhir'])
                return X_tfidf, df['polarity']

            X, y = preprocess_tfidf(clean_df)

                # Label encoding
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state["le"] = le
                # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state["y_test"] = y_test
            # Define Logistic Regression parameters with explicit regularization
            lr_params = {
                'C': [0.001, 0.01, 0.1, 1],  # Smaller C means stronger regularization
                'solver': ['liblinear', 'lbfgs'],  
                'penalty': ['l1', 'l2']  # L1 for sparsity, L2 for generalization (liblinear supports both, lbfgs only L2)
            }

            # Grid Search with Cross-Validation
            gs_lr = GridSearchCV(LogisticRegression(max_iter=500), lr_params, cv=5, scoring='accuracy', n_jobs=-1)
            gs_lr.fit(X_train, y_train)

            # Get the best model
            best_lr = gs_lr.best_estimator_
            st.session_state["best_lr"] = best_lr

            # Predictions
            y_pred_train_lr = best_lr.predict(X_train)
            y_pred_test_lr = best_lr.predict(X_test)
            st.session_state["y_pred_test_lr"] = y_pred_test_lr

            from sklearn.metrics import classification_report, accuracy_score

            # Evaluasi per kategori dan masukkan ke DataFrame
            def evaluate_per_class_df(y_true, y_pred, label_encoder):
                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=label_encoder.classes_,
                    output_dict=True
                )
            
                accuracy = accuracy_score(y_true, y_pred)
            
                # Ambil semua label kelas
                labels = list(label_encoder.classes_)
            
                # Inisialisasi dictionary
                rows = {}
                for label in labels:
                    rows[label.capitalize()] = {
                        "Precision": report[label]["precision"],
                        "Recall": report[label]["recall"],
                        "F1-Score": report[label]["f1-score"]
                    }
            
                # Tambahkan akurasi (hanya 1 nilai di kolom Precision)
                rows["Accuracy"] = {
                    "Precision": accuracy,
                    "Recall": None,
                    "F1-Score": None
                }
            
                # Buat DataFrame
                df_combined = pd.DataFrame.from_dict(rows, orient="index")
                return df_combined

            
            # Panggil fungsi dan simpan ke session state
            df_eval_metrics = evaluate_per_class_df(y_test, y_pred_test_lr, le)
            st.session_state["df_eval_metrics"] = df_eval_metrics
            
            # Tampilkan di Streamlit
            #st.subheader("📊 Evaluasi Performa Model per Kategori")
            #st.dataframe(df_eval_metrics.style.format("{:.2%}", na_rep=""))

        
            # # Evaluate Accuracy
            # accuracy_train_lr = accuracy_score(y_train, y_pred_train_lr)
            # accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)

            #     # Evaluate model
            # def evaluate_model(y_true, y_pred, model_name):
            #     accuracy = accuracy_score(y_true, y_pred)
            #     precision = precision_score(y_true, y_pred, average='weighted')
            #     recall = recall_score(y_true, y_pred, average='weighted')
            #     f1 = f1_score(y_true, y_pred, average='weighted')
            #     return accuracy, precision, recall, f1

            # acc_train_lr, prec_train_lr, rec_train_lr, f1_train_lr = evaluate_model(y_train, y_pred_train_lr, "Logistic Regression (Train)")
            # acc_test_lr, prec_test_lr, rec_test_lr, f1_test_lr = evaluate_model(y_test, y_pred_test_lr, "Logistic Regression (Test)")

            # # Create a container
            
                
            # evaluation_data = {
            #     "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            #     "Training Score": [acc_train_lr, prec_train_lr, rec_train_lr, f1_train_lr],
            #     "Testing Score": [acc_test_lr, prec_test_lr, rec_test_lr, f1_test_lr]
            # }

            # # Convert to DataFrame
            # df_evaluation = pd.DataFrame(evaluation_data)
            # st.session_state["eval_df"] = df_evaluation

        
            # Display in Streamlit
                        #st.write("Cleaned Dataset Shape:", clean_df.shape)
                    #st.write(clean_df.head())  # Show sample cleaned data
        
                    #clean_df.info()
                        
            #     st.download_button(
            #         label="Unduh Ulasan sebagai CSV",
            #         data=csv_bytes,
            #         file_name="ulasan_aplikasi.csv",
            #         mime="text/csv"
            #     )
            # else:
            #     st.warning("CSV belum dibuat. Silakan generate CSV terlebih dahulu.")

        except Exception as e:
            st.write(e)
        
                    


if st.sidebar.button("📊 DataFrame"):
    switch_page("DataFrames")

if st.session_state["current_page"] == "DataFrames": 
    st.markdown(
        """
        <div class="transparent-container">
            <h3>📊 Dataframe</h3>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    try:
        if st.session_state['csv'] is not None:
            #csv_bytes = st.session_state['csv'] #original reviews hanya ulasan tok 
            clean_df = st.session_state["clean_df"].copy()
            csv_bytess = clean_df.to_csv(index=False).encode('utf-8') #keseluruhan df stelah preprocess
            st.download_button(
                label="Download Reviews",
                data=csv_bytess,
                file_name="Application_Reviews.csv",
                mime="text/csv"
            )
        else:
            st.markdown(
        """
        <div class="transparent-container">
            <h5>⚠️You have not done data scraping, please do scraping first.</h5>
        </div>
    
        """,
        unsafe_allow_html=True
    )
            
    
                
          
        
    






    except Exception as e:
        st.write("_")

    
           
                

                # Display processed data
                #st.write("### Processed Data Sample")
                #st.write(clean_df.head())


    
           
            # Sentiment analysis function
            
            
# # Generating WordCloud for negative tweets
#             negative_tweets = clean_df[clean_df['polarity'] == 'negative']
#             list_wordsnegatif = ' '.join(word for tweet in negative_tweets['text_stopword'] for word in tweet)
#             wordcloud = WordCloud(width=600, height=400, background_color='white', min_font_size=10).generate(list_wordsnegatif)

#                 # Displaying WordCloud
#             st.write("### Word Cloud of Negative Tweets Data")
#             fig, ax = plt.subplots(figsize=(8, 6))
#             ax.set_title('Word Cloud of Negative Tweets Data', fontsize=18)
#             ax.grid(False)
#             ax.imshow(wordcloud)
#             fig.tight_layout(pad=0)
#             ax.axis('off')
#             st.pyplot(fig)

#                 # Generating WordCloud for positive tweets
#             positive_tweets = clean_df[clean_df['polarity'] == 'positive']
#             list_wordspositive = ' '.join(word for tweet in positive_tweets['text_stopword'] for word in tweet)
#             wordcloud = WordCloud(width=600, height=400, background_color='white', min_font_size=10).generate(list_wordspositive)

#                 # Displaying WordCloud
#             st.write("### Word Cloud of Positive Tweets Data")
#             fig, ax = plt.subplots(figsize=(8, 6))
#             ax.set_title('Word Cloud of Positive Tweets Data', fontsize=18)
#             ax.grid(False)
#             ax.imshow(wordcloud)
#             fig.tight_layout(pad=0)
#             ax.axis('off')
#             st.pyplot(fig)
            
            
            
            # Cache the sentiment analysis function
    

            

            # Initialize session state for navigation
           

            # Initialize session state for navigation
            # if "current_page" not in st.session_state:
            #     st.session_state["current_page"] = "Vibe Scopium"  # Default page
    try:        
        clean_df = st.session_state["clean_df"].copy()
        if st.session_state.get('do_stemming_choice'== "Yes"):
            # st.dataframe(clean_df[['content', 'score','thumbsUpCount','at','appVersion','text_clean', 'text_casefolding','text_slang_fixed','text_tokenized','text_stopword',
            #                                'text_stemming',
            #                                'text_akhir', 'polarity_score', 'polarity']],use_container_width=True , height=6000) 
            columns_to_show = ['content', 'score','thumbsUpCount','at','appVersion',
                   'text_clean', 'text_casefolding','text_slang_fixed','text_tokenized',
                   'text_stopword', 'text_stemming','text_akhir', 'polarity_score', 'polarity']
           # Let user select the column to filter by
            selected_column = st.selectbox("Choose column to filter by:", columns_to_show)
            
            # Let user input the search value
            search_value = st.text_input(f"Enter value to filter '{selected_column}' by:")
            
            # Filter the DataFrame based on search
            if search_value:
                filtered_df = clean_df[clean_df[selected_column].astype(str).str.contains(search_value, case=False, na=False)]
            else:
                filtered_df = clean_df
            
            # Show the filtered DataFrame but keep all columns
            st.dataframe(filtered_df[columns_to_show], use_container_width=True, height=6000)
           
            
            
             
        else:
            # st.dataframe(clean_df[['content', 'score','thumbsUpCount','at','appVersion','text_clean', 'text_casefolding','text_slang_fixed','text_tokenized','text_stopword',
            #                                 #'text_stemming',
            #                                 'text_akhir', 'polarity_score', 'polarity']],use_container_width=True , height=6000) 
            columns_to_show = ['content', 'score','thumbsUpCount','at','appVersion',
                   'text_clean', 'text_casefolding','text_slang_fixed','text_tokenized',
                   'text_stopword', 'text_akhir', 'polarity_score', 'polarity']
           # Let user select the column to filter by
            selected_column = st.selectbox("Choose column to filter by:", columns_to_show)
            
            # Let user input the search value
            search_value = st.text_input(f"Enter value to filter '{selected_column}' by:")
            
            # Filter the DataFrame based on search
            if search_value:
                filtered_df = clean_df[clean_df[selected_column].astype(str).str.contains(search_value, case=False, na=False)]
            else:
                filtered_df = clean_df
            
            # Show the filtered DataFrame but keep all columns
            st.dataframe(filtered_df[columns_to_show], use_container_width=True, height=6000)

    except Exception as e:
        st.write("_")
    
    
# if st.sidebar.button("Word Cloud"):
#     switch_page("Word Cloud")

# if st.session_state["current_page"] == "Word Cloud":
#     if "clean_df" not in st.session_state or st.session_state["clean_df"] is None:
#         st.warning("You have not done data scraping, please do scraping first.")
#     else:
#         clean_df = st.session_state["clean_df"]
#         pd.set_option('display.max_colwidth', 100000)
        
    

#     # Function to generate WordCloud (cached)
#     @st.cache_data
#     def generate_wordcloud(text_list):
#         text = ' '.join(word for tweet in text_list for word in tweet)  # Flatten list
#         return WordCloud(
#             width=800,  # Smaller width
#             height=400,  # Smaller height
#             background_color='black',  # Set to None for transparency
#             mode="RGBA",
#             colormap="cool",
#             min_font_size=10  # Smaller font size
#         ).generate(text)

#     # Function to display WordCloud (No Title, Smaller Size)
#     def display_glowing_wordcloud(wordcloud):
#         fig, ax = plt.subplots(figsize=(6, 2), dpi=300)  # Higher DPI for better quality
#         fig.patch.set_alpha(0)  # Make figure background transparent
#         ax.set_facecolor("none")  # Transparent background

#         ax.imshow(wordcloud, interpolation="bilinear")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_frame_on(False)
#         ax.axis("off")  # Remove all borders and ticks

#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

#         st.pyplot(fig)


#     # Ensure dataset is available
#     if "clean_df" in st.session_state:
#         clean_df = st.session_state["clean_df"]

#         if "polarity" in clean_df.columns and "text_stopword" in clean_df.columns:
                    
#             # Generate word clouds only once
#             if "wordcloud_neg" not in st.session_state or "wordcloud_pos" not in st.session_state:

#                 # WordCloud for Negative Tweets
#                 negative_tweets = clean_df[clean_df['polarity'] == 'negative']['text_stopword']
#                 if not negative_tweets.empty:
#                     st.session_state["wordcloud_neg"] = generate_wordcloud(negative_tweets)
#                 else:
#                     st.session_state["wordcloud_neg"] = None

#                 # WordCloud for Positive Tweets
#                 positive_tweets = clean_df[clean_df['polarity'] == 'positive']['text_stopword']
#                 if not positive_tweets.empty:
#                     st.session_state["wordcloud_pos"] = generate_wordcloud(positive_tweets)
#                 else:
#                     st.session_state["wordcloud_pos"] = None

#             # Display WordCloud for Negative Tweets (No Title, Smaller)
#             if st.session_state["wordcloud_neg"] is not None:
#                 display_glowing_wordcloud(st.session_state["wordcloud_neg"])
#             else:
#                 st.write("No negative tweets found.")

#             # Display WordCloud for Positive Tweets (No Title, Smaller)
#             if st.session_state["wordcloud_pos"] is not None:
#                 display_glowing_wordcloud(st.session_state["wordcloud_pos"])
#             else:
#                 st.write("No positive tweets found.")

#         else:
#             st.error("Missing 'polarity' or 'text_stopword' column. Ensure sentiment analysis is completed first.")
#     else:
#         st.error("Clean dataset is missing. Please run preprocessing first.")
    



# Sidebar button to toggle the dropdown
if "show_tweets_options" not in st.session_state:
    st.session_state["show_tweets_options"] = False

if st.sidebar.button("😐 Tweets"):
    st.session_state["show_tweets_options"] = not st.session_state["show_tweets_options"]

# Dropdown for selecting tweet sentiment (changes page instantly)
if st.session_state["show_tweets_options"]:
    selected_tweet_page = st.sidebar.radio("Select Tweet Sentiment:", ["😀 Positive", "😡 Negative"], key="tweets_radio")

    # Update current page instantly when selection changes
    if selected_tweet_page and st.session_state["current_page"] != selected_tweet_page:
        st.session_state["current_page"] = selected_tweet_page
        st.experimental_rerun()  # Forces an instant update

# Page content based on current selection
if st.session_state["current_page"] == "😀 Positive":
    st.markdown(
        """
        <div class="transparent-container">
            <h3>😀 Positive Tweets</h3>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    if "clean_df" not in st.session_state or st.session_state["clean_df"] is None:
         st.markdown(
        """
        <div class="transparent-container">
            <h5>⚠️You have not done data scraping, please do scraping first.</h5>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    else:
        clean_df = st.session_state["clean_df"]
        pd.set_option('display.max_colwidth', 100000)
        
    

    # Function to generate WordCloud (cached)
    @st.cache_data
    def generate_wordcloud(text_list):
        excluded_words = {'game'}
        filtered_words = [word for tweet in text_list for word in tweet if word.lower() not in excluded_words]
        text = ' '.join(filtered_words)
        return WordCloud(
            width=800,  # Smaller width
            height=400,  # Smaller height
            background_color='black',  # Set to None for transparency
            mode="RGBA",
            colormap="cool",
            min_font_size=10  # Smaller font size
        ).generate(text)

    # Function to display WordCloud (No Title, Smaller Size)
    def display_glowing_wordcloud(wordcloud):
        fig, ax = plt.subplots(figsize=(6, 2), dpi=300)  # Higher DPI for better quality
        fig.patch.set_alpha(0)  # Make figure background transparent
        ax.set_facecolor("none")  # Transparent background

        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.axis("off")  # Remove all borders and ticks

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

        st.pyplot(fig)

    try:
        # Ensure dataset is available
        if "clean_df" in st.session_state:
            clean_df = st.session_state["clean_df"]
    
            if "polarity" in clean_df.columns and "text_stopword" in clean_df.columns:
                        
                # Generate word clouds only once
                if "wordcloud_neg" not in st.session_state or "wordcloud_pos" not in st.session_state:
    
                    
                    positive_tweets = clean_df[clean_df['polarity'] == 'positive']['text_stopword']
                    word_listpositive = ' '.join(word for tweet in positive_tweets for word in tweet).split()
                    st.session_state["word_listpositive"] = word_listpositive.copy()
                    if not positive_tweets.empty:
                        st.session_state["wordcloud_pos"] = generate_wordcloud(positive_tweets)
                    else:
                        st.session_state["wordcloud_pos"] = None
    
                # Display WordCloud for Positive Tweets (No Title, Smaller)
                if st.session_state["wordcloud_pos"] is not None:
                    display_glowing_wordcloud(st.session_state["wordcloud_pos"])
                else:
                    st.write("No positive tweets found.")
    
            else:
                st.error("Missing 'polarity' or 'text_stopword' column. Ensure sentiment analysis is completed first.")
        else:
            st.error("Clean dataset is missing. Please run preprocessing first.")
    except Exception:
        st.write("_")
    try:
        #st.write("### Most Frequent Words")
        positive_tweets = st.session_state.get("positive_tweets")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(clean_df['text_akhir'])  # Sparse matrix
        word_sums = np.array(X.sum(axis=0)).flatten()
        tfidf_df = pd.DataFrame({'index': vectorizer.get_feature_names_out(), 'jumlah': word_sums})
        tfidf_df = tfidf_df.sort_values('jumlah', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        # sns.barplot(x='jumlah', y='index', data=tfidf_df, ax=ax)
        # ax.set_title('Most Frequent Words')
        # st.pyplot(fig)
                    # Set background color to black

        
        
        # Retrieve word list from session state
        # Retrieve word list from session state
        word_listpositive = st.session_state.get("word_listpositive")
        
        # Make sure the word list is valid
        if word_listpositive and len(word_listpositive) >= 1:
        
            # Selectbox shown later, but declared here to get selected value early
            ngram_size = st.selectbox(
                "🔄 Want to change N-gram size?",
                options=[1, 2, 3, 4, 5],
                index=2,
                help="Choose how many words to combine in a phrase"
            )
        
            # Create dynamic n-grams
            if len(word_listpositive) >= ngram_size:
                ngrams = [' '.join(word_listpositive[i:i+ngram_size]) for i in range(len(word_listpositive) - ngram_size + 1)]
        
                excluded_words = ['game', 'tolong', 'bug', 'perbaiki','lag','frezee','freeze','force','close','sulit']
                ngrams = [gram for gram in ngrams if all(word not in gram.lower() for word in excluded_words)]
        
                # Count and get top 20
                ngram_counts = Counter(ngrams)
                top_ngrams = ngram_counts.most_common(30)
                df_top_ngrams = pd.DataFrame(top_ngrams, columns=['ngram', 'frequency'])
        
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
        
                sns.barplot(x='frequency', y='ngram', data=df_top_ngrams, palette='Greens_r', ax=ax)
        
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00008B")
                    spine.set_linewidth(1)
                    spine.set_alpha(0.7)
        
                ax.set_title(f'Top 30 Most Frequent Positive {ngram_size}-grams', fontsize=16, color="white", weight="bold")
                ax.set_xlabel("Frequency", fontsize=14, color="white")
                ax.set_ylabel("Words", fontsize=14, color="white")
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.grid(color="#000000", linestyle="--", linewidth=1, alpha=0.5)
        
                # Show plot
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ Not enough words to form {ngram_size}-grams.")
        
        else:
            st.warning("⚠️ Word list not available or empty.")





        # fig.patch.set_facecolor('black')
        # ax.set_facecolor('black')
        # word_listpositive = st.session_state.get("word_listpositive")
        # #word_listpositive = ' '.join(word for tweet in positive_tweets['text_stopword'] for word in tweet).split()
    
                    
        # word_counts = Counter(word_listpositive)
        # top_words = word_counts.most_common(20)
        # df_top_words = pd.DataFrame(top_words, columns=['word', 'frequency'])
    
        # fig, ax = plt.subplots(figsize=(12, 6))
    
        #             # Set background color to black
        # fig.patch.set_facecolor('black')
        
        #             # Seaborn barplot
        # sns.barplot(x='frequency', y='word', data=df_top_words, palette='Greens_r', ax=ax)
    
        #             # Apply a glow effect on the borders
        # for spine in ax.spines.values():
        #     spine.set_edgecolor("#00008B")  # Green neon effect
        #     spine.set_linewidth(1)  # Thicker border for glow
        #     spine.set_alpha(0.7)  # Semi-transparent for glow effect
    
        #             # Title and labels
        # ax.set_title('Top 20 Most Frequent Positive Words', fontsize=16, color="white", weight="bold")
        # ax.set_xlabel("Frequency", fontsize=14, color="white")
        # ax.set_ylabel("Words", fontsize=14, color="white")
    
        #             # Change ticks color
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
    
        #             # Apply a glowing effect to grid lines (optional)
        # ax.grid(color="#000000", linestyle="--", linewidth=1, alpha=0.5)
    
        # st.pyplot(fig)

        st.markdown(
            """
            <div class="transparent-container">
                <h5>If the most common word is "keren" 😎, this suggests that users often associate this word with positive experiences. In social media and reviews, "keren" is frequently used to describe something exciting, trendy, or enjoyable. The frequent appearance of this word means that people react enthusiastically to topics they find appealing or engaging.<br><br>Similarly, words like "hebat" 🎉, "cinta" ❤️, and "menakjubkan" 🤩 often appear when users express enthusiasm and satisfaction. These words are common in product reviews, social media posts, and feedback where people want to share strongly positive emotions. If a dataset contains customer reviews, we might also see words like "puas" ✅ or "direkomendasikan" 👍 appearing at the top.<br><br>If "senang" 😊 or "menyenangkan" 🎠 are frequently used, the dataset may contain tweets related to entertainment, joyful experiences, or celebrations. The presence of such words indicates that people often share positive emotions when discussing certain topics.</h5>
            </div>
        
            """,
            unsafe_allow_html=True
        )
                
    #             # Most Frequent Positive Words
    # st.write("### Top 20 Most Frequent Positive Words")
    # word_listpositive = st.session_state.get("word_listpositive")
    # word_counts = Counter(word_listpositive)
    # top_words = word_counts.most_common(20)
    # df_top_words = pd.DataFrame(top_words, columns=['word', 'frequency'])
    # fig, ax = plt.subplots(figsize=(12, 6))
    # sns.barplot(x='frequency', y='word', data=df_top_words, palette='Greens_r', ax=ax)
    # ax.set_title('Top 20 Most Frequent Positive Words')
    # st.pyplot(fig)
    except Exception:
            st.write("_")
    
elif st.session_state["current_page"] == "😡 Negative":
    st.markdown(
        """
        <div class="transparent-container">
            <h3>😡 Negative Tweets</h3>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    if "clean_df" not in st.session_state or st.session_state["clean_df"] is None:
         st.markdown(
        """
        <div class="transparent-container">
            <h5>⚠️You have not done data scraping, please do scraping first.</h5>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    else:
        clean_df = st.session_state["clean_df"]
        pd.set_option('display.max_colwidth', 100000)
        
    

    # Function to generate WordCloud (cached)
    @st.cache_data
    def generate_wordcloud(text_list):
        excluded_words = {'game', 'bagus'}
        filtered_words = [word for tweet in text_list for word in tweet if word.lower() not in excluded_words]
        text = ' '.join(filtered_words)
        return WordCloud(
            width=800,  # Smaller width
            height=400,  # Smaller height
            background_color='black',  # Set to None for transparency
            mode="RGBA",
            colormap="cool",
            min_font_size=10  # Smaller font size
        ).generate(text)

    # Function to display WordCloud (No Title, Smaller Size)
    def display_glowing_wordcloud(wordcloud):
        fig, ax = plt.subplots(figsize=(6, 2), dpi=300)  # Higher DPI for better quality
        fig.patch.set_alpha(0)  # Make figure background transparent
        ax.set_facecolor("none")  # Transparent background

        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.axis("off")  # Remove all borders and ticks

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

        st.pyplot(fig)

    try:
        # Ensure dataset is available
        if "clean_df" in st.session_state:
            clean_df = st.session_state["clean_df"]
    
            if "polarity" in clean_df.columns and "text_stopword" in clean_df.columns:
                        
                # Generate word clouds only once
                if "wordcloud_neg" not in st.session_state or "wordcloud_pos" not in st.session_state:
    
                    
                    negative_tweets = clean_df[clean_df['polarity'] == 'negative']['text_stopword']
                    word_listnegative = ' '.join(word for tweet in negative_tweets for word in tweet).split()
                    st.session_state["word_listnegative"] = word_listnegative.copy()
                    if not negative_tweets.empty:
                        st.session_state["wordcloud_pos"] = generate_wordcloud(negative_tweets)
                    else:
                        st.session_state["wordcloud_pos"] = None
    
                # Display WordCloud for negative Tweets (No Title, Smaller)
                if st.session_state["wordcloud_pos"] is not None:
                    display_glowing_wordcloud(st.session_state["wordcloud_pos"])
                else:
                    st.write("No negative tweets found.")
    
            else:
                st.error("Missing 'polarity' or 'text_stopword' column. Ensure sentiment analysis is completed first.")
        else:
            st.error("Clean dataset is missing. Please run preprocessing first.")
    except Exception:
            st.write("_") 
    try:
        #st.write("### Most Frequent Words")
        negative_tweets = st.session_state.get("negative_tweets")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(clean_df['text_akhir'])  # Sparse matrix
        word_sums = np.array(X.sum(axis=0)).flatten()
        tfidf_df = pd.DataFrame({'index': vectorizer.get_feature_names_out(), 'jumlah': word_sums})
        tfidf_df = tfidf_df.sort_values('jumlah', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        # sns.barplot(x='jumlah', y='index', data=tfidf_df, ax=ax)
        # ax.set_title('Most Frequent Words')
        # st.pyplot(fig)
                    # Set background color to black
        
        # Assuming word_listpositive is already a list of words
        word_listnegative = st.session_state.get("word_listnegative")
        
        # Make sure the word list is valid
        if word_listnegative and len(word_listnegative) >= 1:
        
            # Selectbox shown later, but declared here to get selected value early
            ngram_size = st.selectbox(
                "🔄 Want to change N-gram size?",
                options=[1, 2, 3, 4, 5],
                index=2,
                help="Choose how many words to combine in a phrase"
            )
        
            # Create dynamic n-grams
            if len(word_listnegative) >= ngram_size:
                ngrams = [' '.join(word_listnegative[i:i+ngram_size]) for i in range(len(word_listnegative) - ngram_size + 1)]
        
                excluded_words = ['bagus','game']
                ngrams = [gram for gram in ngrams if all(word not in gram.lower() for word in excluded_words)]
        
                # Count and get top 20
                ngram_counts = Counter(ngrams)
                top_ngrams = ngram_counts.most_common(30)
                df_top_ngrams = pd.DataFrame(top_ngrams, columns=['ngram', 'frequency'])
        
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
        
                sns.barplot(x='frequency', y='ngram', data=df_top_ngrams, palette='Reds_r', ax=ax)
        
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00008B")
                    spine.set_linewidth(1)
                    spine.set_alpha(0.7)
        
                ax.set_title(f'Top 30 Most Frequent Negative {ngram_size}-grams', fontsize=16, color="white", weight="bold")
                ax.set_xlabel("Frequency", fontsize=14, color="white")
                ax.set_ylabel("Words", fontsize=14, color="white")
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.grid(color="#000000", linestyle="--", linewidth=1, alpha=0.5)
        
                # Show plot
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ Not enough words to form {ngram_size}-grams.")
        
        else:
            st.warning("⚠️ Word list not available or empty.")

        
        # fig.patch.set_facecolor('black')
        # ax.set_facecolor('black')
        # word_listnegative = st.session_state.get("word_listnegative")
        # #word_listnegative = ' '.join(word for tweet in negative_tweets['text_stopword'] for word in tweet).split()
    
                    
        # word_counts = Counter(word_listnegative)
        # top_words = word_counts.most_common(20)
        # df_top_words = pd.DataFrame(top_words, columns=['word', 'frequency'])
    
        # fig, ax = plt.subplots(figsize=(12, 6))
    
        #             # Set background color to black
        # fig.patch.set_facecolor('black')
        
        #             # Seaborn barplot
        # sns.barplot(x='frequency', y='word', data=df_top_words, palette='Reds_r', ax=ax)
    
        #             # Apply a glow effect on the borders
        # for spine in ax.spines.values():
        #     spine.set_edgecolor("#00008B")  # Green neon effect
        #     spine.set_linewidth(1)  # Thicker border for glow
        #     spine.set_alpha(0.7)  # Semi-transparent for glow effect
    
        #             # Title and labels
        # ax.set_title('Top 20 Most Frequent Negative Words', fontsize=16, color="white", weight="bold")
        # ax.set_xlabel("Frequency", fontsize=14, color="white")
        # ax.set_ylabel("Words", fontsize=14, color="white")
    
        #             # Change ticks color
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
    
        #             # Apply a glowing effect to grid lines (optional)
        # ax.grid(color="#000000", linestyle="--", linewidth=1, alpha=0.5)
    
        # st.pyplot(fig)
        
        st.markdown(
            """
            <div class="transparent-container">
                <h5>On the other hand, if words like "bad" 👎, "terburuk" 😡, or "mengecewakan" 😞 are among the most common, it indicates negative sentiment. Users tend to use these words when expressing frustration, dissatisfaction, or regret about an experience, product, or event.<br><br>For instance, if "lemot" 🐢 is frequently mentioned, it might be related to complaints about service delays or performance issues. If "rusak" 💔 appears often, it suggests that many users are reporting defects or malfunctions.<br><br>In cases where words like "penipuan" 🚨 or "sampah" 🗑️ are dominant, the dataset could contain reviews about fraudulent activities or poor-quality experiences. These words provide insights into the pain points and common complaints users face.<br><br>By analyzing both positive and negative word patterns, we gain a deeper understanding of how users perceive a given topic. The visualization allows us to identify key themes, track customer satisfaction, and even detect potential issues that need addressing.</h5>
            </div>
        
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.write("_")

if st.sidebar.button("🩻 Evaluation"):
    switch_page("🩻 Evaluation")

if st.session_state["current_page"] == "🩻 Evaluation": 
        # Preprocessing
    with st.container():
                st.markdown(
        """
        <div class="transparent-container">
            <h3>🩻 Model Evaluation</h3>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    try:
            clean_df= st.session_state["clean_df"]
            y_test = st.session_state["y_test"]
            best_lr = st.session_state["best_lr"]
            y_pred_test_lr = st.session_state["y_pred_test_lr"]
            #df_evaluation = st.session_state['eval_df'].copy()
            df_evaluation = st.session_state['df_eval_metrics'].copy()
            le = st.session_state["le"]
            st.dataframe(df_evaluation.style.format(precision=6),use_container_width=True, width=50)  # Formats numbers to 6 decimal places


            import matplotlib.patheffects as path_effects

            # Confusion matrix
            st.markdown(
                """
                <div class="transparent-container">
                    <h3>📊 Confusion Matrix</h3>
                </div>
            
                """,
                unsafe_allow_html=True
            )

            # Create figure
            fig, ax = plt.subplots(figsize=(6,5))  # Adjust size
            cm = confusion_matrix(y_test, y_pred_test_lr)

            # Apply a dark theme
            # Set figure transparency
            fig.patch.set_alpha(0.5)  # 50% transparency
            fig.patch.set_facecolor("#0e0e0e")  # Dark background
            ax.set_facecolor("#0e0e0e")

            # Create heatmap with vibrant colors
            heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='cool', 
                                xticklabels=le.classes_, yticklabels=le.classes_, 
                                ax=ax, cbar=True)  # Enable color bar

            # Modify labels for a glowing effect
            for text in ax.texts:  
                text.set_fontsize(14)  
                text.set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])  # Glow effect

            # Customizing labels
            ax.set_xlabel('Predicted Label', fontsize=12, color="white")
            ax.set_ylabel('True Label', fontsize=12, color="white")
            #ax.set_title('Confusion Matrix - Logistic Regression', fontsize=14, color="cyan")

            # Set X and Y tick colors to white
            ax.tick_params(axis='x', colors='white')  
            ax.tick_params(axis='y', colors='white')  

            # **Modify color bar text color to white**
            colorbar = heatmap.collections[0].colorbar  
            colorbar.ax.yaxis.set_tick_params(color='white')  # Change tick color
            for label in colorbar.ax.get_yticklabels():
                label.set_color("white")  # Change tick text color to white

            # Display in Streamlit
            # Create two columns
            st.pyplot(fig)

            st.markdown(
                """
                <div class="transparent-container">
                    <h3>🗽 Sentiment Polarity Distribution</h3>
                </div>
            
                """,
                unsafe_allow_html=True
            )
        # Hitung jumlah setiap kategori sentimen
            sentiment_counts = clean_df['polarity'].value_counts()

                # Plot data dengan efek glowing dan latar belakang hitam
            fig, ax = plt.subplots(figsize=(12, 6))

                # Set background color to black
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

                # Seaborn barplot
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, 
            palette=sns.color_palette(["#00FFFF", "#ff00ff"]), ax=ax)




                # Apply a glow effect on the borders
            for spine in ax.spines.values():
                spine.set_edgecolor("#00008B")  # Deep blue neon effect
                spine.set_linewidth(1.5)  # Thicker border for glow
                spine.set_alpha(0.7)  # Semi-transparent for glow effect

                # Title and labels
            #ax.set_title("Distribution of Sentiment Polarity", fontsize=16, color="white", weight="bold")
            ax.set_xlabel("Sentiment", fontsize=14, color="white")
            ax.set_ylabel("Count", fontsize=14, color="white")

                # Change ticks color
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

                # Apply a glowing effect to grid lines (optional)
                #ax.grid(color="#000000", linestyle="--", linewidth=1, alpha=0.5)

            st.pyplot(fig)

            st.markdown(
                """
                <div class="transparent-container">
                    <h3>⭐ Ratings</h3>
                </div>
            
                """,
                unsafe_allow_html=True
            )
                        # Hitung distribusi skor rating
            score_counts = clean_df['score'].value_counts().sort_index()
            
            # Buat plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Latar belakang hitam
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Barplot dengan palet warna neon
            sns.barplot(x=score_counts.index, y=score_counts.values,
                        palette=sns.color_palette(["#00FFFF", "#ff00ff", "#39FF14", "#FFA500", "#FF3131"]),
                        ax=ax)
            
            # Glow efek di border
            for spine in ax.spines.values():
                spine.set_edgecolor("#00008B")  # 
                spine.set_linewidth(1.5)
                spine.set_alpha(0.7)
            
            # Judul dan label sumbu
            ax.set_xlabel("Rating Score", fontsize=14, color="white")
            ax.set_ylabel("Count", fontsize=14, color="white")
            
            # Warna ticks
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            
            # Tampilkan plot di Streamlit
            st.pyplot(fig)

                # Menentukan kesimpulan
            if sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0):
                conclusion = "-"
            elif sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
                conclusion = "-"
            else:
                conclusion = "-"

            #st.write(conclusion)
            st.markdown(
            """
            <div class="transparent-container">
                <h5>If the graph shows that positive sentiment polarity is dominant 😊, it suggests that most of the collected text expresses favorable opinions, indicating satisfaction, strong brand loyalty, or positive engagement. This can be beneficial for businesses or entities as it reflects a good reputation, but it's important to check for potential biases in the dataset. <br><br>Conversely, if negative sentiment is more prevalent 😟, it signals widespread dissatisfaction or criticism, which may require further analysis to identify the root causes. While negative feedback can be concerning, it also presents an opportunity for improvement by addressing common complaints and enhancing overall sentiment.</h5>
            </div>
        
            """,
            unsafe_allow_html=True
        )
      
        
            
    except Exception as e:
         st.markdown(
        """
        <div class="transparent-container">
            <h5>⚠️You have not done data scraping, please do scraping first.</h5>
        </div>
    
        """,
        unsafe_allow_html=True
    )
         st.write(e)
    
if st.sidebar.button("🩺 Predict"):
    switch_page("🩺 Predict")

if st.session_state["current_page"] == "🩺 Predict": 
            st.markdown(
                        """
                        <div class="transparent-container">
                            <h3>🩺 Predict</h3>
                        </div>
                    
                        """,
                        unsafe_allow_html=True
                    )
            try:
            #         clean_df= st.session_state["clean_df"]
            #         X = clean_df['text_akhir']
                    
            #         y = clean_df['polarity']
            #         data_size = len(clean_df)
                        
                       
            #         #tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7, ngram_range=(1,2))
        
            #         @st.cache_resource
            #         def preprocess_tfidf(df):
            #             if "tfidf_vectorizer" not in st.session_state or st.session_state.tfidf_vectorizer is None:
            #                 st.session_state.tfidf_vectorizer = TfidfVectorizer(
            #                     max_features=5000, min_df=5, max_df=0.7, ngram_range=(1,2)
            #                 )  
            #                 X_tfidf = st.session_state.tfidf_vectorizer.fit_transform(df['text_akhir'])
            #             else:
            #                 X_tfidf = st.session_state.tfidf_vectorizer.transform(df['text_akhir'])
                    
            #             return X_tfidf, df['polarity']
        
        
            #         X, y = preprocess_tfidf(clean_df)
        
            #             # Label encoding
            #         le = LabelEncoder()
            #         y = le.fit_transform(y)
        
            #             # Split data
            #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
            #         # Define Logistic Regression parameters with explicit regularization
            #         lr_params = {
            #             'C': [0.001, 0.01, 0.1, 1],  # Smaller C means stronger regularization
            #             'solver': ['liblinear', 'lbfgs'],  
            #             'penalty': ['l1', 'l2']  # L1 for sparsity, L2 for generalization (liblinear supports both, lbfgs only L2)
            #         }
        
            #         # Grid Search with Cross-Validation
            #         gs_lr = GridSearchCV(LogisticRegression(max_iter=500), lr_params, cv=5, scoring='accuracy', n_jobs=-1)
            #         gs_lr.fit(X_train, y_train)
        
            #         # Get the best model
            #         best_lr = gs_lr.best_estimator_
        
            #         # Predictions
            #         y_pred_train_lr = best_lr.predict(X_train)
            #         y_pred_test_lr = best_lr.predict(X_test)
        
            #         # Evaluate Accuracy
            #         accuracy_train_lr = accuracy_score(y_train, y_pred_train_lr)
            #         accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)
        
            #             # Evaluate model
            #         def evaluate_model(y_true, y_pred, model_name):
            #             accuracy = accuracy_score(y_true, y_pred)
            #             precision = precision_score(y_true, y_pred, average='weighted')
            #             recall = recall_score(y_true, y_pred, average='weighted')
            #             f1 = f1_score(y_true, y_pred, average='weighted')
            #             return accuracy, precision, recall, f1
        
            #         acc_train_lr, prec_train_lr, rec_train_lr, f1_train_lr = evaluate_model(y_train, y_pred_train_lr, "Logistic Regression (Train)")
            #         acc_test_lr, prec_test_lr, rec_test_lr, f1_test_lr = evaluate_model(y_test, y_pred_test_lr, "Logistic Regression (Test)")
        
            
                    # Simpan hasil prediksi ke dalam session state
                    if "hasil_sentimen" not in st.session_state:
                        st.session_state.hasil_sentimen = None
            
                        # Sentiment Prediction
                    kalimat_baru = st.text_input("Insert Sentences to Predict:")
                    if st.button("Start Predictions"):
                        with st.spinner("predicting"):
                            if kalimat_baru:
                                # Preprocessing
                                # def cleaning_text(text):
                                #     text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
                                #     text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
                                #     text = re.sub(r'RT[\s]', '', text)  # Remove RT
                                #     text = re.sub(r"http\S+", '', text)  # Remove links
                                #     text = re.sub(r'[0-9]+', '', text)  # Remove numbers
                                #     text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
                                #     text = text.replace('\n', ' ')  # Replace new line with space
                                #     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations
                                #     text = text.strip()  # Remove leading and trailing spaces
                                #     return text
        
                                # def case_folding_text(text):
                                #     return text.lower()
        
                                # def tokenizing_text(text):
                                #     return word_tokenize(text)
        
                                # def filtering_text(text):
                                #     list_stopwords = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
                                #     custom_stopwords = {'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy'}
                                #     list_stopwords.update(custom_stopwords)
                                #     return [word for word in text if word not in list_stopwords]
        
                                # def stemming_text(text):
                                #     factory = StemmerFactory()
                                #     stemmer = factory.create_stemmer()
                                #     return ' '.join(stemmer.stem(word) for word in text.split())
        
                                # def to_sentence(list_words):
                                #     return ' '.join(list_words)
                                # slang_words = {"@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}
        
                                # def fix_slang_words(text):
                                #     return ' '.join(slang_words.get(word.lower(), word) for word in text.split())
                                kalimat_baru_cleaned = cleaning_text(kalimat_baru)
                                kalimat_baru_casefolded = case_folding_text(kalimat_baru_cleaned)
                                kalimat_baru_slangfixed = fix_slang_words(kalimat_baru_casefolded)
                                kalimat_baru_tokenized = tokenizing_text(kalimat_baru_slangfixed)
                                kalimat_baru_filtered = filtering_text(kalimat_baru_tokenized)
                                kalimat_baru_stemmered = stemming_text(kalimat_baru_filtered)
                                kalimat_baru_final = to_sentence(kalimat_baru_stemmered)
        
                                    # Predict
                                best_lr = st.session_state["best_lr"]
                                tfidf_vectorizer = st.session_state.tfidf_vectorizer
                                X_kalimat_baru = tfidf_vectorizer.transform([kalimat_baru_final]).toarray()
                                prediksi_sentimen = best_lr.predict(X_kalimat_baru)
                                hasil_sentimen = "POSITIF" if prediksi_sentimen[0] == 1 else "NEGATIF"
                                    # Simpan hasil ke session state
                                st.session_state.hasil_sentimen = hasil_sentimen
                                # Tampilkan hasil dengan emoji
                                if hasil_sentimen == "POSITIF":
                                    st.write(f"😃 The sentiment of the sentences is **Positive**.")
                                else:
                                    st.write(f"😡 The sentiment of the sentences is **Negative**.")
                    # if st.session_state.hasil_sentimen:
                    #     st.write(f"🔹 Sentimen kalimat baru adalah **{st.session_state.hasil_sentimen}**.")
                    st.markdown(
                        """
                        <div class="transparent-container">
                            <h5>Example of positive or negative sentiment:
                
                _
                
                Saya suka ini! Kualitasnya luar biasa, dan melebihi ekspektasi saya.
                
                _
                
                Ini adalah pengalaman terburuk yang pernah saya alami. Banyak sekali bug dan error di dalamnya!
                </h5>
                        </div>
                    
                        """,
                        unsafe_allow_html=True
                    )
            except Exception:
                        st.markdown(
                        """
                        <div class="transparent-container">
                            <h5>⚠️You have not done data scraping, please do scraping first.</h5>
                        </div>
                    
                        """,
                        unsafe_allow_html=True
                    )
if st.sidebar.button("👨‍✈️ About Me"):
    switch_page("👨‍✈️ About Me")

if st.session_state["current_page"] == "👨‍✈️ About Me": 
    st.markdown(
        """
        <div class="transparent-container">
            <h1>👨‍✈️ About Me</h1>
            <h4>Hi, I’m Fauzan Fadhillah Arisandi!<br><br>Right now, I’m working on my final project to complete my degree. I’m pursuing a bachelor’s degree, but for this last assignment, I need to create something useful for the community. That’s why I’m building VibeScopium 🚀—a sentiment analysis app that determines whether a review is positive or negative. The cool part? It focuses only on reviews from Indonesia 🇮🇩.<br><br>VibeScopium is powered by TF-IDF, Logistic Regression, and a lexicon-based approach 🤖📊. Basically, it converts text into numerical values using TF-IDF (Term Frequency-Inverse Document Frequency) to figure out which words are important, and then Logistic Regression steps in to classify whether a review is positive or negative. Additionally, it also incorporates a lexicon-based approach tailored for Indonesian sentiment analysis, allowing the app to recognize sentiment-heavy words commonly used in local reviews. This combination makes the analysis fast, efficient, and accurate! If an app is flooded with bad reviews, developers can quickly identify issues and improve their products. On the flip side, positive sentiment means an app is doing great! 🎉<br><br>I’ve always been passionate about data science, and Python is my favorite language 🐍. VibeScopium is still a work in progress, and I’m constantly fine-tuning the model to make it even better. Working on this project has been a game-changer for me—I’ve learned so much about machine learning, NLP, and model optimization. It’s challenging, but honestly, I love every second of it. 🔥<br><br>Moving forward, I want to dive even deeper into data science and build more impactful projects. For now, my main focus is making VibeScopium as accurate and user-friendly as possible. Hopefully, it’ll become a valuable tool for developers, businesses, and anyone looking to understand Play Store reviews in Indonesia! 🚀  
</h4>

    
        """,
        unsafe_allow_html=True
    )

            # Page Content
#st.title(st.session_state["current_page"])  # Display the selected page title

            # Specific Page Contents
# if st.session_state["current_page"] == "DataFrames":
#     st.write("## Cleaned Data")

#     if "clean_df" in st.session_state:
#         st.write(st.session_state["clean_df"].head())  
#     else:
#         st.error("Clean dataset is missing. Please run preprocessing first.")

            # Other pages (Blank)
#elif st.session_state["current_page"] in ["Vibe Scopium", "Input App ID", "Word Cloud", "Evaluation", "Predict", "About Me"]:
    #st.write(f"### This is the {st.session_state['current_page']} page.")
    #st.write("Content coming soon!")







#st.markdown('<h1 class="h1">Hello, Streamlit!</h1>', unsafe_allow_html=True)
#st.header("Exploring Sentiments, Unveiling Insights")

# Initialize session state if not set




            
           
