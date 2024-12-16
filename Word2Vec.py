import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """Honourable Governor of Uttar Pradesh, Anandiben Patel Ji, Honourable Chief Minister Shri Yogi Adityanath Ji, Honourable Deputy Chief Ministers Keshav Prasad Maurya Ji and Brajesh Pathak Ji, esteemed ministers, respected Members of Parliament and Legislative Assembly, the Mayor and District Panchayat President of Prayagraj, other distinguished guests, and my dear brothers and sisters, I bow with reverence to this sacred land of Sangam in Prayagraj and extend my obeisance to all the saints and sages arriving for the Maha Kumbh. I commend the tireless efforts of the employees, workers, and sanitation staff ensuring the success of the event. Hosting such a grand event, welcoming lakhs of devotees daily, and conducting a Maha Yagya for 45 days are all contributing to the creation of a new chapter in Prayagraj’s history. This event will elevate our nation’s cultural and spiritual identity. Bharat is a land of sacred rivers and holy places, with Prayag being of unparalleled spiritual significance. It is where all divine powers converge, as described in our scriptures. Prayag is not only the meeting point of three rivers but also offers all four goals of life—Dharma, Artha, Kama, and Moksha. I consider it a blessing to visit Prayag repeatedly and perform a snan at the Sangam. Projects worth thousands of crores, including the Hanuman and Akshaya Vat Corridors, have been inaugurated in connection with the event. The Maha Kumbh is a living testament to India’s unbroken cultural and spiritual journey, bringing together people from all walks of life. Historically, saints and spiritual leaders have made decisions here that have greatly influenced the nation’s history. Previous governments failed to recognize the importance of the Kumbh, but today, both the central and state governments are ensuring seamless facilities for devotees. Connectivity improvements and initiatives worth crores are a testament to our commitment to the event. The Ramayana Circuit, the Shri Krishna Circuit, and the grand Ram Temple in Ayodhya reflect our government’s vision to preserve heritage while fostering development. In Prayagraj, the Akshaya Vat Corridor and other sacred sites are being renovated. Prayagraj, the land of Nishadraj, is also home to Shringaverpur Dham, a symbol of the divine bond between Lord Ram and Nishadraj. Cleanliness is essential for the success of the Kumbh, and sanitation workers play a crucial role in maintaining it. The Maha Kumbh also boosts economic activities, creating employment opportunities and contributing to economic growth. The event integrates technology, with the Kumbh Sahayak Chatbot and other innovations combining tradition and modernity. As Bharat progresses toward becoming a developed nation, the spiritual energy of the Maha Kumbh will strengthen our resolve. May the confluence of Maa Ganga, Maa Yamuna, and Maa Saraswati bring welfare to humanity. I extend my best wishes to every devotee visiting Prayagraj. Bharat Mata Ki Jai! Ganga Mata Ki Jai!"""


# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)
               
# Preparing the dataset
sentences = nltk.sent_tokenize(text)


sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)

words = model.wv.key_to_index


# Finding Word Vectors
vector = model.wv['lord']


# Most similar words
similar = model.wv.most_similar('people')
