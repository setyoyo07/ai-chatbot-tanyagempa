import os
import requests

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

#Function Call to get earthquake data from BMKG
@tool
def get_latest_felt_earthquake() -> dict:
    """Get data from meteorological agency (BMKG) about latest earthquake in Indonesia that felt by people

    Returns:
        dict: A JSON object containing an array of recent earthquake records.
            The structure is defined as follows:
            {
                "Infogempa": {
                    "gempa": [
                        {
                            "Tanggal": str,          # Date of the earthquake (e.g., "14 Jun 2026")
                            "Jam": str,              # Local time of occurrence in WIB (e.g., "17:14:39 WIB")
                            "DateTime": str,         # ISO 8601 standardized UTC timestamp (e.g., "2026-06-14T10:14:39+00:00")
                            "Coordinates": str,      # Latitude and Longitude separated by a comma (e.g., "4.91,96.11")
                            "Lintang": str,          # Latitude coordinate position (e.g., "4.91 LU" or "6.98 LS")
                            "Bujur": str,            # Longitude coordinate position (e.g., "96.11 BT")
                            "Magnitude": str,        # Earthquake magnitude value represented as a string (e.g., "5.6")
                            "Kedalaman": str,        # Depth of the earthquake epicenter (e.g., "10 km")
                            "Wilayah": str,          # Description of the epicenter location or distance to the nearest landmark/city
                            "Potensi": str,          # [Optional] Tsunami potential status, usually present in M >= 5.0 events (e.g., "Tidak berpotensi tsunami")
                            "Dirasakan": str         # [Optional] MMI intensity scale and list of areas that felt the tremor (e.g., "II-III Sukabumi, II Cianjur")
                        }
                    ]
                }
            }
    """
    url = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"

    json_data = {}
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
    
    return json_data

@tool
def get_latest_earthquake_above_5_mag() -> dict:
    """Get data from meteorological agency (BMKG) about latest significant earthquake in Indonesia (has magnitude >= 5.0)

    Returns:
        dict: A JSON object containing an array of recent significant earthquake records.
            The structure is defined as follows:
            
            {
                "Infogempa": {
                    "gempa": [
                        {
                            "Tanggal": str,          # Date of the earthquake (e.g., "13 Jun 2026")
                            "Jam": str,              # Local time of occurrence in WIB (e.g., "19:05:35 WIB")
                            "DateTime": str,         # ISO 8601 standardized UTC timestamp (e.g., "2026-06-13T12:05:35+00:00")
                            "Coordinates": str,      # Latitude and Longitude coordinates separated by a comma (e.g., "1.10,126.20")
                            "Lintang": str,          # Latitude coordinate position (e.g., "1.10 LU")
                            "Bujur": str,            # Longitude coordinate position (e.g., "126.20 BT")
                            "Magnitude": str,        # Earthquake magnitude scale represented as a string (e.g., "5.1")
                            "Kedalaman": str,        # Depth of the earthquake epicenter (e.g., "10 km")
                            "Wilayah": str,          # Distance and direction description from the nearest city/island epicenter (e.g., "125 km Tenggara BITUNG-SULUT")
                            "Potensi": str           # Tsunami potential threat status declaration (e.g., "Tidak berpotensi tsunami")
                        }
                    ]
                }
            }
    """
    url = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"

    json_data = {}
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
    
    return json_data

# Streamlit code for chatbot UI
st.set_page_config(
    page_title="TanyaGempa - BMKG AI Assistant",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Kustomisasi Avatar untuk Chat Bubbles
AVATAR_USER = "👤"
AVATAR_AI = "🤖"

with st.sidebar:
    st.title("Info & Edukasi")
    st.markdown("---")
    st.info(
        "💡 **Tips Bertanya:**\n"
        "- *'Apakah ada gempa di Jawa Barat baru-baru ini?'*\n"
        "- *'Info gempa Aceh sore tadi dong'*",
        icon="ℹ️"
    )
    st.markdown("---")
    st.caption("Data bersumber langsung dari real-time JSON API TEWS - BMKG Indonesia.")

st.title("🌐 TanyaGempa")
st.subheader("Asisten AI Informasi Gempa Bumi Terkini")
st.markdown("---")

# Not run app if no API Key provided
if (
    os.environ.get("GROQ_API_KEY", "") == ""
):
    st.stop()

# Client LLM creation
client = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
tools_list = [get_latest_earthquake_above_5_mag, get_latest_felt_earthquake]
tools_map = {
    "get_latest_felt_earthquake": get_latest_felt_earthquake,
    "get_latest_earthquake_above_5_mag": get_latest_earthquake_above_5_mag
}
client = client.bind_tools(tools_list)

# Chat history init
system_intruction_prompt = """
You are a professional and friendly BMKG (Indonesian Meteorological, Climatological, and Geophysical Agency) officer. 
Your job is to answer questions about the latest earthquakes in Indonesia using the provided tools.

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. When a user asks about an earthquake in a specific region or a specific time (e.g., "Aceh", "sore tadi", "tadi malam"), you MUST CALL BOTH TOOLS: `get_latest_earthquake_above_5_mag` AND `get_latest_felt_earthquake`.
2. Do NOT assume the earthquake data is only in one tool. Even if the user says "gempa terkini", the earthquake might be small but felt by people (M < 5.0), meaning it is only recorded in `get_latest_felt_earthquake`.
3. Analyze the data from BOTH tools carefully before formulating your response.

RESPONSE GUIDELINES:
- If the earthquake is found in `get_latest_felt_earthquake`, mention the areas where it was felt (skala MMI / "Dirasakan").
- If the earthquake is found in `get_latest_earthquake_above_5_mag`, mention its tsunami potential status ("Potensi").
- If after checking BOTH tools, no matching earthquake data is found for that specific region/time, politely inform the user: "Berdasarkan data resmi BMKG terbaru, tidak ada catatan gempa di wilayah [Nama Wilayah] untuk waktu tersebut."
- Always reply in a polite, informative manner using Indonesian. Keep your answers clear, structuring the earthquake details (Magnitude, Location, Depth, Time) using bullet points if multiple data found.
- If the user's question is completely unrelated to earthquakes, politely refuse and state that you can only assist with earthquake information.
"""
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [SystemMessage(system_intruction_prompt)]

# Render chat history
for chat in st.session_state["chat_history"]:
    if isinstance(chat, HumanMessage):
        with st.chat_message("human", avatar=AVATAR_USER):
            st.markdown(chat.content)
    elif isinstance(chat, SystemMessage) or isinstance(chat, ToolMessage):
        continue
    elif hasattr(chat, 'content') and not chat.content.strip():
        continue
    else:
        with st.chat_message("ai", avatar=AVATAR_AI):
            st.markdown(chat.content)

# User Input/Question
user_input = st.chat_input("Chat here")
if not user_input:
    st.stop()

# Add user input to chat history
st.session_state["chat_history"].append(HumanMessage(user_input))
with st.chat_message("human", avatar=AVATAR_USER):
    st.markdown(user_input)

# Run LLM
response = client.invoke(st.session_state["chat_history"])
st.session_state["chat_history"].append(response)

# Check if LLM use tool_calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        with st.spinner(f"🔄 Memeriksa database {tool_name}..."):
            selected_tool = tools_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
        st.session_state["chat_history"].append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_id)
        )

    # Get final answer from tools data
    final_response = client.invoke(st.session_state["chat_history"])
    st.session_state["chat_history"].append(final_response)

    with st.chat_message("ai", avatar=AVATAR_AI):
        st.markdown(final_response.content)
else:
    # If LMM no need use tools just show the text
    with st.chat_message("ai", avatar=AVATAR_AI):
        st.markdown(response.content)