# restuarant_ai_agent
# 🍽️ Restaurant AI Agent

An AI-powered restaurant assistant built with **Streamlit**, **Together AI**, and **ElevenLabs**.  
It can take **voice or chat orders**, confirm them, and save them to an Excel sheet (`orders.xlsx`).

---

## 🚀 Features
- Chat and Voice modes for taking orders
- Order parsing (Qty, Item, Price, Customer name, Size)
- Saves orders to Excel (`orders.xlsx`)
- LLM integration via Together AI
- TTS responses using ElevenLabs (with gTTS fallback)

---

## 📦 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/vikshittindwani/restuarant_ai_agent.git
cd restuarant_ai_agent
pip install -r requirements.txt
```
🔑 API Keys

Add your API keys in .streamlit/secrets.toml:
```bash

TOGETHER_API_KEY = "your_together_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
```
▶️ Run the app
```bash
streamlit run os.py
```
## 📂 Output

 >Orders are saved in orders.xlsx in the project directory.
 >You can also download the file directly from the sidebar.

## 🛠️ Notes

If you don’t have an ElevenLabs key, the app will fallback to gTTS for text-to-speech.

For microphone support, ensure pyaudio is installed correctly (pip install pyaudio).

!<img width="1920" height="1080" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/9c6930e0-c115-4b55-abbc-d23b06ad2758" />
!<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/4ae558e3-00f6-43c8-89c5-b215f28bb02b" />
!<img width="1920" height="1080" alt="Screenshot (74)" src="https://github.com/user-attachments/assets/9679da76-a2ed-4f8c-9224-2c329bc00e28" />



