from audiorecorder import audiorecorder
import streamlit as st
import requests
import time
import os

STT_DATA_DIR = "/data/ai-nlp-stt-service/exp/tmp"
STT_AUDIO_FILE = f"{STT_DATA_DIR}/audio.wav"

if not os.path.exists(STT_DATA_DIR):
    os.mkdir(STT_DATA_DIR)

def convert_audio_to_wav(audio_path, save_path, is_mono=True):
    ac = 1 if is_mono else 2

    os.system(
        "ffmpeg -loglevel panic -y -i {} -ac {} -acodec pcm_s16le -ar 16000 {}".format(
            audio_path,
            ac,
            save_path
            )
        )
    
def run_stt(audio_filepath):
    url = "http://localhost:7532/api/infer"

    payload = {}
    files=[
        ('audio', (os.path.basename(audio_filepath), open(audio_filepath, 'rb'), 'audio/wav'))
    ]
    headers = {
        'AUTHORIZATION': 'PREP a0967870c50b53af66ca698f7df459e9'
    }

    response = requests.request(
        "POST", url, 
        headers=headers, 
        data=payload, 
        files=files
    )
    return response.text

def get_stt_result(audio_path):
    result = run_stt(audio_path)
    return result
    
def record_audio():
    audio = audiorecorder("Click to record", "Click to stop recording")
    if not audio.empty():
        st.audio(audio.export().read())  

        audio.export(os.path.join(STT_DATA_DIR, 'temp'), format="wav")
        convert_audio_to_wav(os.path.join(STT_DATA_DIR, 'temp'), STT_AUDIO_FILE)
        
def upload_audio():
    uploaded_file = st.file_uploader("→ Choose a audio", type=['wav'])
    if uploaded_file is not None:
        with open(os.path.join(STT_DATA_DIR, 'temp'), "wb") as f:
            f.write(uploaded_file.getbuffer())
        convert_audio_to_wav(os.path.join(STT_DATA_DIR, 'temp'), STT_AUDIO_FILE)
        st.audio(STT_AUDIO_FILE)

def input_script():
    script = st.text_area('→ Enter the script here *(optional)*', height=120)
    return script

def submit(container: st.container):
    with container:
        with st.spinner('Please **:red[WAIT]** for some seconds ...'):       
            time_start = time.time()            
            st.session_state['prep_stt_result'] = get_stt_result(
                audio_path=STT_AUDIO_FILE,
            )            
            st.session_state['prep_stt_time'] = round((time.time() - time_start), 2)                

def create_side_bar():
    st.sidebar.write('⚙️ <i>Select audio source</i>', unsafe_allow_html=True)
    st.session_state['audio_src'] = st.sidebar.radio(
        "→ Set audio source",
        key="visibility",
        options=["upload", "record"],
        horizontal=True,
        label_visibility='collapsed'
    )

    st.sidebar.write('---')
    

if __name__ == '__main__':
    st.sidebar.write('---')

    st.title(":blue[G-Group] Speech2Text")

    # audio source
    if 'audio_src' not in st.session_state:
        st.session_state['audio_src'] = ''
    # stt result
    if 'prep_stt_result' not in st.session_state:
        st.session_state['prep_stt_result'] = None
    # stt time
    if 'prep_stt_time' not in st.session_state:
        st.session_state['prep_stt_time'] = 0

    # TODO: create side bar
    create_side_bar()

    # set audio source
    if st.session_state['audio_src'] == "upload":
        upload_audio()
    else:
        record_audio()

    # input script
    script = input_script()

    # create container
    container = st.container()

    with container:
        st.write('---')
        st.markdown(
            f'<b><span style="color:red; font-family: Times New Roman; font-size: 25px; background-color: papayawhip">Speech2Text:</span></b>', 
            unsafe_allow_html=True
        )
        submit = st.button(
            '🚀 **:red[Submit]**', 
            key='prep', 
            on_click=submit, 
            kwargs=dict(container=container)
        )
        if st.session_state['prep_stt_result'] is not None:
            st.write(
                '***:blue[→ Transcript]***: {}'.format(
                    st.session_state['prep_stt_result']
                    )
                )
            st.success(
                '🎉 **:red[DONE]** in {} seconds'.format(
                    st.session_state['prep_stt_time']
                    )
                )
                
