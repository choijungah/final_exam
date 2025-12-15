import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import altair as alt
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx

st.set_page_config(
    page_title="Kpop Demon Hunters Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('C011208 최정아의 K팝 데몬 헌터스 팬덤 분석 보고서')
st.divider()

st.sidebar.header('📌 분석 옵션')
st.sidebar.markdown('### 시각화 설정')

viz_option = st.sidebar.selectbox(
    '분석 유형을 선택하세요',
    ['전체 개요', '워드클라우드', '네트워크 분석', '시간대별 패턴', '키워드 빈도', '날짜별 추이']
)

@st.cache_data
def load_data():
    df = pd.read_csv('kpop_demon_hunters_news.csv')
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    return df

df = load_data()

show_data = st.sidebar.checkbox('원본 데이터 보기', value=False)

if show_data:
    st.subheader('📋 수집된 뉴스 데이터')
    st.dataframe(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("총 기사 수", len(df))
    col2.metric("수집 기간", f"{df['pubDate'].min().date()} ~ {df['pubDate'].max().date()}")
    col3.metric("평균 일일 기사 수", f"{len(df) / (df['pubDate'].max() - df['pubDate'].min()).days:.1f}")
    st.divider()

@st.cache_data
def preprocess_text(df):
    from konlpy.tag import Okt
    okt = Okt()
    with open('stop_str.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    all_nouns = []
    for text in df['description'].tolist():
        text_cleaned = re.sub(r'[^가-힣\s]', '', str(text))
        nouns = okt.nouns(text_cleaned)
        nouns = [word for word in set(nouns) if (len(word) > 1) and (word not in stopwords)]
        all_nouns.append(nouns)
    return all_nouns

with st.spinner('데이터 전처리 중...'):
    all_nouns = preprocess_text(df)

if viz_option == '전체 개요':
    st.header('📊 팬덤 형성 핵심 요인 종합 분석')
    st.markdown('''
    ### 분석 목적
    - 케이팝 데몬 헌터스에 대한 온라인 뉴스 데이터 분석
    - 팬덤 형성의 핵심 요인을 다각도로 분석
    - 데이터 기반 인사이트 제공
    ''')
    st.info('왼쪽 사이드바에서 분석 유형을 선택하세요', icon="ℹ️")
    st.subheader('기본 통계')
    col1, col2 = st.columns(2)
    with col1:
        st.write('#### 날짜별 기사 개수')
        df['date'] = df['pubDate'].dt.date
        date_counts = df.groupby('date').size().reset_index(name='count')
        st.line_chart(date_counts.set_index('date'))
    with col2:
        st.write('#### 시간대별 기사 개수')
        df['hour'] = df['pubDate'].dt.hour
        hour_counts = df.groupby('hour').size()
        st.bar_chart(hour_counts)

elif viz_option == '워드클라우드':
    st.header('☁️ 워드클라우드 분석')
    max_words = st.sidebar.slider('표시할 최대 단어 수', min_value=20, max_value=100, value=50, step=10)

    import os
    if os.path.exists('NanumGothic.ttf'):
        han_font_path = 'NanumGothic.ttf'
    elif os.path.exists('C:\\Windows\\Fonts\\malgun.ttf'):
        han_font_path = 'C:\\Windows\\Fonts\\malgun.ttf'
    else:
        han_font_path = None

    text = ' '.join([word for nouns in all_nouns for word in nouns])
    wc = WordCloud(font_path=han_font_path, max_words=max_words, width=800, height=800, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title('케이팝 데몬헌터스 주요 키워드', size=20)
    st.pyplot(fig)
    st.success(f'총 {max_words}개의 키워드가 표시되었습니다', icon="✅")

elif viz_option == '네트워크 분석':
    st.header('🕸️ 키워드 네트워크 분석')
    min_count = st.sidebar.number_input('최소 연결 빈도', min_value=5, max_value=50, value=20, step=5)
    edge_list = []
    for nouns in all_nouns:
        if len(nouns) > 1:
            edge_list.extend(combinations(sorted(nouns), 2))
    edge_counts = Counter(edge_list)
    filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight >= min_count}
    if len(filtered_edges) == 0:
        st.warning(f'최소 빈도 {min_count} 이상인 연결이 없습니다. 값을 낮춰보세요.', icon="⚠️")
    else:
        G = nx.Graph()
        weighted_edges = [(node1, node2, weight) for (node1, node2), weight in filtered_edges.items()]
        G.add_weighted_edges_from(weighted_edges)
        pos_spring = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        edge_widths = [G[u][v]['weight'] * 0.05 for u, v in G.edges()]
        fig, ax = plt.subplots(figsize=(15, 15))

        import os
        import matplotlib.font_manager as fm
        if os.path.exists('NanumGothic.ttf'):
            font_prop = fm.FontProperties(fname='NanumGothic.ttf')
            font_name = font_prop.get_name()
        else:
            font_name = 'sans-serif'

        nx.draw_networkx(G, pos_spring, with_labels=True, node_size=node_sizes, width=edge_widths, font_family=font_name, font_size=12, node_color='skyblue', edge_color='gray', alpha=0.8, ax=ax)
        ax.set_title('케이팝 데몬헌터스 키워드 네트워크', size=20)
        ax.axis('off')
        st.pyplot(fig)
        st.info(f'총 {len(G.nodes())}개의 키워드, {len(G.edges())}개의 연결이 표시되었습니다', icon="ℹ️")

elif viz_option == '시간대별 패턴':
    st.header('🕐 요일별/시간대별 뉴스 발행 패턴')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_days = st.sidebar.multiselect('표시할 요일 선택', day_order, default=day_order)
    df['day_of_week'] = df['pubDate'].dt.day_name()
    df['hour'] = df['pubDate'].dt.hour
    df_filtered = df[df['day_of_week'].isin(selected_days)]
    heatmap_data = df_filtered.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in selected_days])
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(data=heatmap_pivot, annot=True, cmap="coolwarm", fmt=".0f", linewidths=.5, linecolor='black', ax=ax)
    ax.set_title("요일별/시간대별 뉴스 발행 패턴", size=16)
    ax.set_xlabel("시간대", size=12)
    ax.set_ylabel("요일", size=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.success('히트맵을 통해 뉴스가 가장 많이 발행되는 시간대를 확인할 수 있습니다', icon="✅")

elif viz_option == '키워드 빈도':
    st.header('📊 상위 키워드 빈도 분석')
    chart_type = st.sidebar.radio('차트 유형 선택', ['막대 그래프', '수평 막대 그래프'])
    all_words = [word for nouns in all_nouns for word in nouns]
    word_counts = Counter(all_words)
    top_20 = word_counts.most_common(20)
    df_keywords = pd.DataFrame(top_20, columns=['keyword', 'count'])
    if chart_type == '막대 그래프':
        chart = alt.Chart(df_keywords).mark_bar().encode(x=alt.X('keyword:N', sort='-y', title='키워드'), y=alt.Y('count:Q', title='빈도수'), color=alt.Color('count:Q', scale=alt.Scale(scheme='blues')), tooltip=['keyword', 'count']).properties(height=400, width=700, title='상위 20개 키워드 빈도')
    else:
        chart = alt.Chart(df_keywords).mark_bar().encode(y=alt.Y('keyword:N', sort='-x', title='키워드'), x=alt.X('count:Q', title='빈도수'), color=alt.Color('count:Q', scale=alt.Scale(scheme='greens')), tooltip=['keyword', 'count']).properties(height=500, width=700, title='상위 20개 키워드 빈도')
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df_keywords)

elif viz_option == '날짜별 추이':
    st.header('📈 날짜별 뉴스 기사 개수 추이')
    df['date'] = df['pubDate'].dt.date
    date_counts = df.groupby('date').size().reset_index(name='count')
    fig = px.line(date_counts, x="date", y="count", markers=True, width=900, height=500, labels={'count': '기사 개수', 'date': '날짜'}, title="날짜별 뉴스 기사 개수 추이")
    st.plotly_chart(fig, key="date_trend", on_select="rerun")
    st.markdown('''
    ### 인사이트
    - 특정 날짜에 급증한 구간은 주요 이벤트(신규 에피소드 공개 등)와 연관
    - 지속적인 언급량은 팬덤의 지속적인 관심도를 나타냄
    ''')

st.divider()
st.caption('케이팝 데몬헌터스 팬덤 분석 대시보드 - 데이터 시각화 최종 프로젝트')
