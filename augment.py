import streamlit as st
from parrot import Parrot
import pandas as pd
import torch
import warnings
import os
import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings("ignore")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.image('./healthee-logo.png')

col1, mid, col2 = st.columns([1, 1, 20])
with col1:
    st.image('./chef.jpeg', width=60)
with col2:
    st.header("l'augmentateur")

uploaded_file = st.file_uploader("Choose a file", type={"csv", "txt"})

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    cols = st.sidebar.selectbox('select columns containing user data:', df.columns.tolist())

    do_diverse = st.sidebar.checkbox('Diversity', value=True,
                                     help='Should paraphrases should be different than the original?')
    reproducibility = st.sidebar.checkbox('Reproducibility', value=True,
                                          help='Whether results should be fixed on each run')
    fluency = st.sidebar.slider('Fluency', min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                help='To what degree should the paraphrases be grammatically correct')
    adequacy = st.sidebar.slider('Adequacy', min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                 help='To what degree should the paraphrases keep the original meaning')


def random_state(seed: object) -> object:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


all_paraphrases = []


def run():
    st.write('Please wait, heating up the stove')
    if reproducibility:
        random_state(1234)

    if cols:
        questions_list = df[cols].to_list()

    with st.spinner('Initializing augmentation recipe...'):
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    bar = st.progress(0)

    for i, question in enumerate(questions_list):
        with st.spinner('cooking...'):
            paraphrases = parrot.augment(input_phrase=question,
                                         use_gpu=False,
                                         do_diverse=do_diverse,
                                         fluency_threshold=fluency,
                                         adequacy_threshold=adequacy)
        all_paraphrases.append(paraphrases)
        st.markdown(f'Augmentation for: *"{question}*"')
        para_df = pd.DataFrame(paraphrases)
        para_df.columns = ['sentence', 'score']
        st.write(para_df)
        bar.progress(int(i / len(questions_list) * 100))

    st.balloons()


if uploaded_file and cols:
    st.button('Augment!', on_click=run)

if len(all_paraphrases) > 0:
    st.sidebar.download_button('Export all paraphrases',
                               pd.DataFrame(all_paraphrases).to_csv().encode('utf-8'),
                               file_name=str(round(datetime.datetime.timestamp(datetime.datetime.now()))) +
                                         '_augmentation_output.csv')
