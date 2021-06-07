# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Jun, 2021
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
    apptitle = 'Mass-spring-damping-analysis'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Dimension reduction for mass-spring harmonic oscillator')

    st.markdown('''
    ## Objectives

    1. Illustrate the motion of the mass with time captured by camera.
    2. Reduce the dimension of the camera data and identify the intrinsic dimension by principal component analysis (PCA).
    3. Identify governing physical parameters for the spring mass system (natural frequency, mass, spring constant, damping, etc.
    ''')

    # Experimental set
    st.markdown('## Experimental set')
    st.image('src/schematic.png')

    # Dataset
    st.markdown('## Dataset')

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

    st.markdown('## Raw data')
    st.dataframe(df)

    lines = ['1', '2', '3']
    chosen_line = st.selectbox('Select a camera to show raw data', lines, 0)
    global ratio
    ratio = st.slider('Select a ratio to show raw data', 0.01, 1.0, 1.0)

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        video_file = open(f'src/Camera{chosen_line}.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    col1, col2, col3 = st.beta_columns(3)
    show_length = int(df.shape[0] * ratio)-1
    with col1:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'x{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=14)
        plt.ylabel(f'x{chosen_line}', fontsize=14)
        plt.title(f'x{chosen_line}', fontsize=16)
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = plt.figure()
        plt.plot(df['t'][:show_length], df[f'y{chosen_line}'][:show_length])
        plt.xlabel('t', fontsize=14)
        plt.ylabel(f'y{chosen_line}', fontsize=14)
        plt.title(f'y{chosen_line}', fontsize=16)
        st.pyplot(fig, clear_figure=True)
    with col3:
        fig = plt.figure()
        plt.scatter(df[f'x{chosen_line}'][:show_length], df[f'y{chosen_line}'][:show_length])
        plt.xlabel(f'x{chosen_line}', fontsize=14)
        plt.ylabel(f'y{chosen_line}', fontsize=14)
        plt.title('x-y', fontsize=16)
        st.pyplot(fig, clear_figure=True)

    st.markdown('Some statistics:')
    st.dataframe(df.describe())

    # basic information for this dataset
    df_info = df.describe()

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

    st.markdown('## PCA')
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
    st.markdown(f'Eigenvalues: {eigenvalues}')

    x_major_locator = MultipleLocator(1)  # used to set x-axis tick interval

    fig = plt.figure(figsize=(10, 4))

    # plot the log of eigenvalues
    ax1 = fig.add_subplot(1, 2, 1)
    # ax1.semilogy(range(1, 7), s, '-o', color='k')
    ax1.plot(range(1, 7), eigenvalues, '-o', color='k')
    ax1.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax1.set_xlabel('Principal Component Number', fontsize=16)
    ax1.set_ylabel('Eigenvalues', fontsize=16)

    # plot the cumulative sum ratio of eigenvalues
    energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    # energy = np.cumsum(s) / np.sum(s)
    # # compare energy with sklearn
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=6)
    # pca.fit(df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3']])
    # print(np.cumsum((pca.explained_variance_ratio_)))
    # print(np.cumsum(eigenvalues)/np.sum(eigenvalues))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, 7), energy, '-o', color='k')
    ax2.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax2.set_xlabel('Principal Component Number', fontsize=16)
    ax2.set_ylabel('Cumulative Energy', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown('''
    The i-th Cumulative Energy $e_{i}$ is defined as: 

    $e_{i}=\\frac{\\sum_{k=1}^{i}\\lambda_k}{\\sum_{k=1}^{m}\\lambda_k}$

    , where $m$ is the total number of eigenvalues, $\\lambda$ is an eigenvalue.
    ''')

    threshold = st.slider('Select a threshold for energy:', 0.8, 1.0, 0.9)
    real_dimension = 0
    for idx, e_i in enumerate(energy.tolist()):
        if e_i > threshold:
            real_dimension = idx+1
            break

    st.markdown(f'''
        **This is a {real_dimension} dimensional problem.**
    ''')

    # fig = plt.figure(figsize=(12, 8))
    # for i in range(1, 7):
    #     fig.add_subplot(2, 3, i)
    #     plt.plot(df['t'], Y[i-1, :])
    #     plt.title('Principal Componenet {}'.format(i))
    # plt.tight_layout()
    # st.pyplot(fig, clear_figure=True)

    st.markdown('## Compute covariance matrix of transformed data')

    # covariance matrix
    C_Y = np.cov(Y)
    C_B = np.cov(B)
    np.set_printoptions(suppress=True)

    st.write(C_Y)

    st.markdown('''A spring-mass system will oscillate in a pattern that matches a sine wave. 
    If the system has some damping, the sine wave can be multiplied by an exponential function
    $z(t)=A*sin⁡(2πft)*exp⁡(-bt)$
    where A is the starting amplitude, f is the natural frequency in Hz, and b is the damping coefficient. 
    The natural frequency is the inverse of the peak-to-peak distance of the sine wave. 
    The damping coefficient is computed by considering the rate of the exponential decay of the sine wave.
    ''')

    st.image('src/pic2.png', caption='Reduced 1D data by PCA with damped sinusoidal plot overlaid.')
    st.markdown('Estimated amplitude, natural frequency and damping coefficient: (To be developed).')
if __name__ == '__main__':
    main()
