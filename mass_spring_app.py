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

    # Experimental set
    st.markdown('## Experimental set')
    col1, col2 = st.beta_columns([1, 1])
    with col1:
        st.image('src/1.jpg', width=300)
    with col2:
        st.image('src/2.jpg', width=250)

    st.markdown('## Raw data')
    st.dataframe(df)

    lines = ['a', 'b', 'c']
    chosen_line = st.selectbox('Select a camera to show raw data', lines, 0)
    global ratio
    ratio = st.slider('Select a ratio to show raw data', 0.01, 1.0, 1.0)

    col1, col2 = st.beta_columns(2)
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

    st.markdown('## Substract Mean Value')
    df_part = df[['t', 'xa_new', 'ya_new',
                  'xb_new', 'yb_new', 'xc_new', 'yc_new']]
    st.dataframe(df_part)

    lines2 = ['a', 'b', 'c']
    chosen_line2 = st.selectbox(
        'Select a camera to show parsed data', lines2, 0)
    global ratio2
    ratio2 = st.slider('Select a ratio to show parsed data', 0.01, 1.0, 1.0)

    col1, col2 = st.beta_columns(2)
    show_length = int(df_part.shape[0] * ratio2)-1
    with col1:
        fig = plt.figure()
        plt.plot(df_part['t'][:show_length],
                 df_part[f'x{chosen_line2}_new'][:show_length])
        plt.xlabel('t', fontsize=14)
        plt.ylabel(f'x{chosen_line2}_new', fontsize=14)
        plt.title(f'x{chosen_line2}_new', fontsize=16)
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = plt.figure()
        plt.plot(df_part['t'][:show_length],
                 df_part[f'y{chosen_line2}_new'][:show_length])
        plt.xlabel('t', fontsize=14)
        plt.ylabel(f'y{chosen_line2}_new', fontsize=14)
        plt.title(f'y{chosen_line2}_new', fontsize=16)
        st.pyplot(fig, clear_figure=True)

    st.markdown('Some statistics:')
    st.dataframe(df_part.describe())

    # original data: m*N (m: data point number, N: dimension for each data)
    X = df[['xa', 'ya', 'xb', 'yb', 'xc', 'yc']].to_numpy()

    # transpose X as B
    B = df[['xa_new', 'ya_new', 'xb_new',
            'yb_new', 'xc_new', 'yc_new']].to_numpy().T

    # SVD
    U, s, VT = svd(B, full_matrices=0)

    # Transform B to Y
    Y = U.T.dot(B)

    st.markdown('## SVD')
    # st.markdown('''
    # ```
    # # original data: m*N (m: data point number, N: dimension for each data)
    # X = df[['xa', 'ya', 'xb', 'yb', 'xc', 'yc']].to_numpy()

    # # transpose X as B
    # B = df[['xa_new', 'ya_new', 'xb_new', 'yb_new', 'xc_new', 'yc_new']].to_numpy().T

    # # SVD
    # U, s, VT = svd(B, full_matrices=0)
    # ```
    # ''')
    st.markdown(f'Singular Values: {s}')

    x_major_locator = MultipleLocator(1)  # used to set x-axis tick interval

    fig = plt.figure(figsize=(10, 4))

    # plot the log of eigenvalues
    ax1 = fig.add_subplot(1, 2, 1)
    # ax1.semilogy(range(1, 7), s, '-o', color='k')
    ax1.plot(range(1, 7), s, '-o', color='k')
    ax1.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax1.set_xlabel('Principal Component Number', fontsize=16)
    ax1.set_ylabel('Singular Values', fontsize=16)

    # plot the cumulative sum ratio of eigenvalues
    s_squared = s**2
    energy = np.cumsum(s_squared) / np.sum(s_squared)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, 7), energy, '-o', color='k')
    ax2.xaxis.set_major_locator(x_major_locator)  # set x-axis tick interval
    ax2.set_xlabel('Principal Component Number', fontsize=16)
    ax2.set_ylabel('Cumulative Energy', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


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

    st.markdown('`C_Y = np.cov(Y)`')
    st.write(C_Y)

    st.markdown('`C_B = np.cov(B)`')
    st.write(C_B)

    # st.markdown('Based on the diagonals of the covariance matrix $C_Y$, we can see that the first singular value is far more than other values, which means the first raw has large variance than other rows. Thus, the measure system is intrinsically one-dimensional.')


if __name__ == '__main__':
    main()
