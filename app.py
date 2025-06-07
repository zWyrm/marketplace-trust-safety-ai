import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import math
import re

st.set_page_config(
    page_title="Amazon Product and Review Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 0rem;
        background-color: white;
    }
    
    .stApp {
        background-color: white;
    }
    
    .product-box {
        padding: 20px;
        margin: 15px 0;
        background: white;
        transition: transform 0.2s;
    }
    
    .product-box:hover {
        transform: translateY(-3px);
    }
    
    .product-image-container {
        text-align: center;
        margin-bottom: 15px;
    }

    /* Fixed button styling with higher specificity */
    div.stButton > button,
    button[data-baseweb="button"],
    .stButton > button:first-child,
    div[data-testid="stButton"] button,
    .stButton button {
        background: white !important;
        border: 1px solid #ccc !important;
        border-radius: 6px !important;
        padding: 10px !important;
        margin: 15px 0 10px 0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #333 !important;
        text-align: center !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    div.stButton > button:hover,
    button[data-baseweb="button"]:hover,
    .stButton > button:first-child:hover,
    div[data-testid="stButton"] button:hover,
    .stButton button:hover {
        background: #f8f9fa !important;
        border-color: #999 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        color: #333 !important;
    }
    
    .price-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    
    .price {
        font-size: 20px;
        font-weight: bold;
        color: #B12704;
    }
    
    .original-price {
        font-size: 14px;
        color: #666;
        text-decoration: line-through;
    }
    
    .discount {
        font-size: 13px;
        color: #cc6600;
        font-weight: bold;
    }
    
    .rating {
        color: #ff9900;
        font-size: 14px;
        margin: 8px 0;
    }
    
    .rating-count {
        color: #0066c0;
        font-size: 13px;
        margin-left: 5px;
    }
    
    .ai-confidence {
        background: linear-gradient(135deg, #e7f3ff 0%, #f0f8ff 100%);
        border: 1px solid #0066c0;
        border-radius: 6px;
        padding: 10px;
        margin-top: 15px;
        font-size: 13px;
        font-weight: 500;
        color: #0066c0;
        text-align: center;
    }
    
    .review-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    .reviewer-name {
        font-weight: bold;
        color: #0066c0;
        font-size: 15px;
        margin-bottom: 8px;
    }
    
    .review-title {
        font-weight: 600;
        font-size: 16px;
        color: #111111;
        margin: 8px 0;
        line-height: 1.3;
    }
    
    .review-content {
        line-height: 1.6;
        color: #111111;
        font-size: 14px;
        margin: 12px 0;
    }
    
    .review-ai-confidence {
        background: #f8f9fa;
        border: 1px solid #0066c0;
        border-radius: 5px;
        padding: 8px;
        margin-top: 12px;
        font-size: 12px;
        font-weight: 500;
        color: #0066c0;
        text-align: center;
    }
    
    .header {
        background: linear-gradient(135deg, #232f3e 0%, #1a252f 100%);
        color: white;
        padding: 20px 0;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 30px 0;
        gap: 10px;
    }
    
    .search-container {
        margin: 20px 0;
    }
    
    .product-detail-container {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .product-detail-title {
        color: #111111 !important;
        font-weight: 600 !important;
        font-size: 28px !important;
        line-height: 1.3 !important;
        margin-bottom: 15px !important;
    }
    
    .product-about-section {
        color: #111111 !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    .product-about-title {
        color: #111111 !important;
        font-weight: 600 !important;
        font-size: 20px !important;
        margin-bottom: 10px !important;
    }

    /* Remove empty containers */
    .element-container:empty {
        display: none !important;
    }

    .stMarkdown:empty {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    ## Load the analysis results and product confidence data with exact decimal preservation
    try:
        analysis_df = pd.read_csv('analysis_results.csv', dtype={'authenticity_score': str})
        product_confidence_df = pd.read_csv('product_confidence.csv')
        return analysis_df, product_confidence_df
    except FileNotFoundError:
        st.error("Data files not found.")
        return None, None

def format_number(value):
    ## Format number with commas, handling both string and numeric inputs
    if pd.isna(value):
        return "0"
    
    if isinstance(value, str):
        cleaned = ''.join(c for c in value if c.isdigit() or c in ',.')
        return cleaned if cleaned else "0"
    
    try:
        return f"{int(float(value)):,}"
    except (ValueError, TypeError):
        return str(value)

def render_stars(rating):
    try:
        rating_float = float(rating)
        full_stars = int(rating_float)
        half_star = 1 if rating_float - full_stars >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        stars = "★" * full_stars + "☆" * half_star + "☆" * empty_stars
        return f'{stars} {rating_float:.1f}'
    except (ValueError, TypeError):
        return '☆☆☆☆☆ N/A'

def process_review_content(content):
    ## to render image URLs
    if pd.isna(content):
        return ""
    
    image_url_pattern = r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^\s]*)?'
    
    def replace_with_image(match):
        url = match.group(0)
        return f'<br><img src="{url}" style="max-width: 300px; max-height: 200px; object-fit: contain; margin: 10px 0; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" /><br>'
    
    processed_content = re.sub(image_url_pattern, replace_with_image, str(content), flags=re.IGNORECASE)
    
    return processed_content

def render_product_card(product_data, product_confidence):
    product_id = product_data['product_id'].iloc[0]
    product_name = product_data['product_name'].iloc[0]
    discounted_price = product_data['discounted_price'].iloc[0]
    actual_price = product_data['actual_price'].iloc[0]
    discount_percentage = product_data['discount_percentage'].iloc[0]
    rating = product_data['rating'].iloc[0]
    rating_count = product_data['rating_count'].iloc[0]
    img_link = product_data['img_link'].iloc[0]
    
    confidence = product_confidence.get(product_id, 75.0)
    
    st.markdown(
        f'<div class="product-box"><div class="product-image-container">',
        unsafe_allow_html=True
    )
    
    if pd.notna(img_link):
        try:
            st.image(img_link, width=180)
        except:
            st.markdown('<div style="width: 180px; height: 180px; background: #f5f5f5; display: flex; align-items: center; justify-content: center; color: #999;">Image unavailable</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="width: 180px; height: 180px; background: #f5f5f5; display: flex; align-items: center; justify-content: center; color: #999;">No image</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # product title
    title_display = product_name[:80] + "..." if len(str(product_name)) > 80 else str(product_name)
    
    if st.button(title_display, key=f"title_{product_id}", use_container_width=True):
        st.session_state.selected_product = product_id
        st.rerun()
    
    # rating 
    rating_html = render_stars(rating)
    formatted_rating_count = format_number(rating_count)
    st.markdown(f'<div class="rating">{rating_html}<span class="rating-count">({formatted_rating_count} ratings)</span></div>', unsafe_allow_html=True)
    
    # price
    try:
        price_display = f"₹{float(discounted_price):,.0f}" if pd.notna(discounted_price) else "Price N/A"
        original_price_display = ""
        discount_display = ""
        
        if pd.notna(actual_price) and float(actual_price) > float(discounted_price):
            original_price_display = f'<span class="original-price">₹{float(actual_price):,.0f}</span>'
        
        if pd.notna(discount_percentage) and float(discount_percentage) > 0:
            discount_display = f'<span class="discount">({float(discount_percentage):.0f}% off)</span>'
        
        price_html = f'''
        <div class="price-container">
            <span class="price">{price_display}</span>
            {original_price_display}
            {discount_display}
        </div>
        '''
        st.markdown(price_html, unsafe_allow_html=True)
    except (ValueError, TypeError):
        st.markdown(f'<div class="price">Price: {discounted_price}</div>', unsafe_allow_html=True)
    
    # AI Confidence
    st.markdown(f'''
    <div class="ai-confidence">
        AI suggests this product is authentic with {confidence:.2f}% confidence
    </div>
    </div>
    ''', unsafe_allow_html=True)

def render_review_card(review_row):
    """Render a single review card in a box with exact authenticity score from CSV"""
    reviewer_name = review_row['user_name']
    review_title = review_row['review_title']
    review_content = review_row['review_content']
    authenticity_score = review_row['authenticity_score']
    
    processed_content = process_review_content(review_content)
    
    if pd.isna(authenticity_score) or str(authenticity_score).strip() in ['', 'nan', 'NaN']:
        exact_score = "N/A"
    else:
        exact_score = str(authenticity_score).strip()
        if '.' in exact_score and exact_score.replace('.', '').replace('-', '').isdigit():
            exact_score = exact_score.rstrip('0').rstrip('.')
    
    review_html = f'''
    <div class="review-box">
        <div class="reviewer-name">{reviewer_name}</div>
        <div class="review-title">{review_title}</div>
        <div class="review-content">{processed_content}</div>
        <div class="review-ai-confidence">
            AI suggests this review is authentic with {exact_score}% confidence
        </div>
    </div>
    '''
    
    st.markdown(review_html, unsafe_allow_html=True)

## pagination 
def paginate_products(products_df, page_size=20):
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    total_products = len(products_df)
    total_pages = math.ceil(total_products / page_size)
    
    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_products = products_df.iloc[start_idx:end_idx]
    
    return paginated_products, total_pages

def render_pagination(current_page, total_pages):
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("First", disabled=(current_page <= 1)):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("Previous", disabled=(current_page <= 1)):
            st.session_state.current_page = current_page - 1
            st.rerun()
    
    with col3:
        st.markdown(f'<div style="text-align: center; padding: 8px; color: #666; font-size: 14px;">Page {current_page} of {total_pages}</div>', unsafe_allow_html=True)
    
    with col4:
        if st.button("Next", disabled=(current_page >= total_pages)):
            st.session_state.current_page = current_page + 1
            st.rerun()
    
    with col5:
        if st.button("Last", disabled=(current_page >= total_pages)):
            st.session_state.current_page = total_pages
            st.rerun()

def main():
    # Load data
    analysis_df, product_confidence_df = load_data()
    
    if analysis_df is None or product_confidence_df is None:
        return
    
    # creating product confidence dictionary
    product_confidence_dict = dict(zip(product_confidence_df['product_id'], 
                                     product_confidence_df['product_confidence']))
    
    # Header
    st.markdown('''
    <div class="header">
        <h1 style="margin: 0; color: white; font-size: 36px;">Amazon AI-Powered Product & Review Analysis</h1>
        <p style="margin: 10px 0 0 0; color: #ccc; font-size: 16px;">Shop with confidence - discover real products and honest reviews backed by AI authenticity scores.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Main content
    if st.session_state.selected_product is None:
        # search bar
        search_term = st.text_input("", placeholder="Search products and categories...")
        
        # reset page when searching
        if 'last_search' not in st.session_state:
            st.session_state.last_search = ""
        
        if search_term != st.session_state.last_search:
            st.session_state.current_page = 1
            st.session_state.last_search = search_term
        
        # filter products based on search
        if search_term:
            filtered_df = analysis_df[
                analysis_df['product_name'].str.contains(search_term, case=False, na=False) |
                analysis_df['category'].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_df = analysis_df
        
        # get unique products
        unique_products = filtered_df.groupby('product_id').first().reset_index()
        
        if len(unique_products) == 0:
            st.write("No products found matching your search.")
            return
        
        # paginate products
        paginated_products, total_pages = paginate_products(unique_products, page_size=20)
        
        cols_per_row = 2
        for i in range(0, len(paginated_products), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(paginated_products):
                    with col:
                        product_data = filtered_df[filtered_df['product_id'] == paginated_products.iloc[i + j]['product_id']]
                        render_product_card(product_data, product_confidence_dict)
        
        # Render pagination
        if total_pages > 1:
            st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
            render_pagination(st.session_state.current_page, total_pages)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Product detail page with reviews
        product_id = st.session_state.selected_product
        product_reviews = analysis_df[analysis_df['product_id'] == product_id]
        
        if len(product_reviews) == 0:
            st.error("Product not found!")
            if st.button("Back to Products"):
                st.session_state.selected_product = None
                st.rerun()
            return
        
        # back button
        if st.button("Back to Products", key="back_button"):
            st.session_state.selected_product = None
            st.rerun()
        
        # Product header - Fixed the empty container issue
        product_info = product_reviews.iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if pd.notna(product_info['img_link']):
                try:
                    st.image(product_info['img_link'], width=300)
                except:
                    st.write("Image unavailable")
            else:
                st.write("No image")
        
        with col2:
            # product title
            st.markdown(f'<h1 class="product-detail-title">{product_info["product_name"]}</h1>', unsafe_allow_html=True)
            st.markdown(f'<div class="rating">{render_stars(product_info["rating"])}</div>', unsafe_allow_html=True)
            formatted_rating_count = format_number(product_info['rating_count'])
            st.write(f"({formatted_rating_count} ratings)")
            
            # price 
            try:
                price_display = f"₹{float(product_info['discounted_price']):,.0f}"
                original_price_display = ""
                discount_display = ""
                
                if float(product_info['actual_price']) > float(product_info['discounted_price']):
                    original_price_display = f'<span class="original-price">₹{float(product_info["actual_price"]):,.0f}</span>'
                
                if float(product_info['discount_percentage']) > 0:
                    discount_display = f'<span class="discount">Save {float(product_info["discount_percentage"]):.0f}%</span>'
                
                price_html = f'''
                <div class="price-container">
                    <span class="price">{price_display}</span>
                    {original_price_display}
                    {discount_display}
                </div>
                '''
                st.markdown(price_html, unsafe_allow_html=True)
                
            except (ValueError, TypeError):
                st.markdown(f'**Price:** {product_info["discounted_price"]}')
            
            # Product AI Confidence
            product_confidence = product_confidence_dict.get(product_id, 75.0)
            st.markdown(f'''
            <div class="ai-confidence">
                AI suggests this product is authentic with {product_confidence:.2f}% confidence
            </div>
            ''', unsafe_allow_html=True)
        
        # Product description
        if pd.notna(product_info['about_product']):
            st.markdown('<h2 class="product-about-title">About this product</h2>', unsafe_allow_html=True)
            about_items = str(product_info['about_product']).split('|')
            for item in about_items:
                if item.strip():
                    st.markdown(f'<div class="product-about-section">• {item.strip()}</div>', unsafe_allow_html=True)
        
        # Reviews section
        st.subheader(f"Customer Reviews ({len(product_reviews)} reviews)")
        
        # sorting reviews by desc authenticity score 
        try:
            product_reviews_sorted = product_reviews.copy()
            product_reviews_sorted['auth_score_numeric'] = pd.to_numeric(product_reviews_sorted['authenticity_score'], errors='coerce')
            product_reviews_sorted = product_reviews_sorted.sort_values('auth_score_numeric', ascending=False, na_last=True)
        except:
            product_reviews_sorted = product_reviews
        
        for idx, (_, review) in enumerate(product_reviews_sorted.iterrows()):
            render_review_card(review)

if __name__ == "__main__":
    main()
