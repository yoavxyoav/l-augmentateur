import streamlit as st
from parrot import Parrot
import pandas as pd
import torch
import warnings
import os
import numexpr
from datetime import datetime

numexpr.set_num_threads(numexpr.detect_number_of_cores())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings("ignore")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

clicked = None

##########
# Header #
##########

header = st.container()
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)
header.image('./healthee-logo.png')

col1, mid, col2 = header.columns([1, 1, 20])
with col1:
    st.image('./chef.jpeg', width=60)
with col2:
    st.header("l'augmentateur")

#########
# Knobs #
#########
do_diverse = st.sidebar.checkbox('Diversity', value=True,
                                 help='Should paraphrases should be different than the original?')
reproducibility = st.sidebar.checkbox('Reproducibility', value=True,
                                      help='Whether container should be fixed on each run')
fluency = st.sidebar.slider('Fluency', min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                            help='To what degree should the paraphrases be grammatically correct')
adequacy = st.sidebar.slider('Adequacy', min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                             help='To what degree should the paraphrases keep the original meaning')

########
# Data #
########

data = st.container()
uploaded_file = data.file_uploader("Choose a file", type={"csv", "txt"})
textbox = data.empty()

user_text = textbox.text_input('or input your own sentence', help='Minimum length must be 5 characters')


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    parrot_model = Parrot(model_tag="./parrot_paraphraser_on_T5/")
    return parrot_model


results = st.container()

with st.spinner('Initializing augmentation recipe...'):
    start = datetime.now()
    parrot = load_model()
    duration = (datetime.now() - start).seconds
    results.info(f'Paraphrasing engine loaded and ready')

if uploaded_file is not None:
    textbox.empty()
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
        data.write(df)
        cols = st.sidebar.selectbox('select columns containing user data:', df.columns.tolist())
        question_list = df[cols].to_list()
    elif uploaded_file.name.endswith('txt'):
        question_list = uploaded_file.read()
        question_list = question_list.decode().split('\n')
        df = pd.DataFrame(question_list)
        data.write(df)
        cols = True
else:
    question_list = [user_text]


def random_state(seed: object) -> object:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


all_paraphrases = []


def run(container):
    container.write('Please wait, heating up the stove')
    if reproducibility:
        random_state(1234)

    bar = container.progress(0)

    for i, question in enumerate(question_list):
        with st.spinner('cooking...'):
            paraphrases = parrot.augment(input_phrase=question,
                                         use_gpu=False,
                                         do_diverse=do_diverse,
                                         fluency_threshold=fluency,
                                         adequacy_threshold=adequacy)
        all_paraphrases.append(paraphrases)
        container.markdown(f'Augmentation for: *"{question}*"')
        para_df = pd.DataFrame(paraphrases)
        try:
            para_df.columns = ['sentence', 'score']
        except:
            pass
        container.write(para_df)
        bar.progress(int(i / len(question_list) * 100))
    bar.progress(100)
    st.balloons()


button = st.empty()
if uploaded_file or len(user_text) > 4:
    clicked = button.button('Augment!')
    if clicked:
        button.empty()
        run(results)
        if len(all_paraphrases) > 0:
            results.download_button('Export all paraphrases',
                                    pd.DataFrame(pd.concat([pd.DataFrame(x) for x in all_paraphrases])).to_csv().encode('utf-8'),
                                    file_name=str(round(datetime.timestamp(datetime.now()))) +
                                              '_augmentation_output.csv')
