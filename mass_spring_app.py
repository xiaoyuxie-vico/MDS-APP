# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Sep, 2021
'''

import io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
from scipy.linalg import svd
import streamlit as st


np.set_printoptions(precision=2)

matplotlib.use('agg')


def main():
    apptitle = 'Mass-Spring-Damper System Analysis'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Dimension reduction for spring-mass-damping system')

    # level 1 font
    st.markdown("""
        <style>
        .L1 {
            font-size:40px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # level 2 font
    st.markdown("""
        <style>
        .L2 {
            font-size:20px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    #########################Objectives#########################

    st.markdown('<p class="L1">Objectives</p>', unsafe_allow_html=True)

    str_1 = '''1. Illustrate the motion of the mass with time captured by camera.'''
    str_2 = '''2. Reduce the dimension of the camera data and identify the intrinsic dimension by principal component analysis (PCA).'''
    # str_3 = '''3. Identify governing physical parameters for the spring mass system (natural frequency, mass, spring constant, damping, etc.'''

    st.markdown('<p class="L2">{}</p>'.format(str_1), unsafe_allow_html=True)
    st.markdown('<p class="L2">{}</p>'.format(str_2), unsafe_allow_html=True)
    # st.markdown('<p class="L2">{}</p>'.format(str_3), unsafe_allow_html=True)

    #########################Experimental set#########################

    # Experimental set

    st.markdown('<p class="L1">Experimental set</p>', unsafe_allow_html=True)
    st.image('src/schematic.png')

    st.markdown('<p class="L2">Videos:</p>', unsafe_allow_html=True)
    str_1 = """[1. Camera 1](https://drive.google.com/file/d/1-BukVXmKl5G-hmR5v1dCUw7tEUy0MXAl/view?usp=sharing)"""
    st.markdown(str_1)
    str_2 = """[2. Camera 2](https://drive.google.com/file/d/1qTe8MC7yvCOlFx2JDDX6W5rvmHKfRCYI/view?usp=sharing)"""
    st.markdown(str_2)
    str_3 = """[3. Camera 3](https://drive.google.com/file/d/1NNwmqHz6wA6kToc6Ydqvs_1Mb7yZjvZX/view?usp=sharing)"""
    st.markdown(str_3)

    #########################Dataset#########################
    st.markdown('<p class="L1">Upload motion data from different cameras</p>', unsafe_allow_html=True)

    flag = ['New dataset', 'Default dataset']
    use_new_data = st.selectbox('Chosse a new dataset or use default dataset', flag, 1)

    # load dataset
    if use_new_data == 'New dataset':
        uploaded_file = st.file_uploader('Choose a CSV file', accept_multiple_files=False)
    
    # button_dataset = st.button('Click once you have selected a dataset')
    # if button_dataset:
    # load dataset
    if use_new_data == 'New dataset':
        data = io.BytesIO(uploaded_file.getbuffer())
        df = pd.read_csv(data)
    elif use_new_data == 'Default dataset':
        file_path = 'src/multidata2.csv'
        df = pd.read_csv(file_path)

    lines = ['Camera 1', 'Camera 2', 'Camera 3']
    chosen_line = st.selectbox('Select a camera to show raw data', lines, 0)
    chosen_line = chosen_line.replace('Camera ', '')
    global ratio
    ratio = st.slider('Select a ratio to show raw data', 0.01, 1.0, 1.0)

    col1, col2, col3 = st.beta_columns(3)
    show_length = int(df.shape[0] * ratio)-1
    with col1:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'x{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=30)
        plt.ylabel(f'x', fontsize=30)
        plt.title(f'x (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'y{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=30)
        plt.ylabel(f'y', fontsize=30)
        plt.title(f'y (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with col3:
        fig = plt.figure()
        plt.scatter(df[f'x{chosen_line}'][:show_length], df[f'y{chosen_line}'][:show_length])
        plt.xlabel(f'x', fontsize=30)
        plt.ylabel(f'y', fontsize=30)
        plt.title(f'x-y (Camera {chosen_line})', fontsize=34)
        plt.tick_params(labelsize=26)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # st.markdown('Some statistics:')
    # st.dataframe(df.describe())

    # basic information for this dataset
    df_info = df.describe()

    #########################PCA#########################
    # subtract mean for each column
    for col_name in df.columns:
        if col_name == 't':
            continue
        df['{}_new'.format(col_name)] = df[col_name] - \
            df_info[col_name]['mean']

    df_part = df[['t', 'x1_new', 'y1_new',
                  'x2_new', 'y2_new', 'x3_new', 'y3_new']]

    # original data: m*N (m: data point number, N: dimension for each data)
    X = df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3']].to_numpy()

    # transpose X as B
    B = df[['x1_new', 'y1_new', 'x2_new',
            'y2_new', 'x3_new', 'y3_new']].to_numpy().T

    # SVD
    U, s, VT = svd(B, full_matrices=0)

    # Transform B to Y
    Y = U.T.dot(B)

    st.markdown('<p class="L1">Dimension reduction by principal component analysis (PCA)</p>', unsafe_allow_html=True)

    # st.markdown('## Principal component analysis (PCA)')
    # st.markdown('''
    # ```
    # # original data: m*N (m: data point number, N: dimension for each data)
    # X = df[['xa', 'ya', 'xb', 'yb', 'xc', 'yc']].to_numpy()

    # # transpose X as B
    # B = df[['x1_new', 'y1_new', 'x2_new', 'y2_new', 'x3_new', 'y3_new']].to_numpy().T

    # # SVD
    # U, s, VT = svd(B, full_matrices=0)
    # ```
    # ''')
    eigenvalues = s**2/(df.shape[0]-1)
    eigenvalues_str = ', '.join([str(round(i, 2)) for i in eigenvalues.tolist()])
    st.markdown('<p class="L2">Eigenvalues:</p>', unsafe_allow_html=True)
    st.markdown(f'[{eigenvalues_str}]')

    x_major_locator = MultipleLocator(1)  # used to set x-axis tick interval

    # plot the log of eigenvalues
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1.semilogy(range(1, 7), s, '-o', color='k')
    ax1.plot(range(1, 7), eigenvalues, '-o', color='k')
    ax1.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax1.set_xlabel('Principal Component Number', fontsize=16)
    ax1.set_ylabel('Eigenvalues', fontsize=16)
    plt.tight_layout()
    plt.grid()
    st.pyplot(fig, clear_figure=True)

    st.markdown('## **Conclusion**: this system is one-dimensional.</p>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
