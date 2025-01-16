def defect_analysis_animation():
    # Professional spinner frames
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Technical process messages relevant to RAG and defect analysis
    words = [
        "🔍 Retrieving Defect Records",
        "📊 Processing Historical Data",
        "🔄 Analyzing Patterns",
        "📈 Computing Similarity Scores",
        "🎯 Matching Similar Cases",
        "📑 Generating Insights"
    ]
    
    spinner = cycle(frames)
    i = 0
    word_idx = 0
    
    while not future.done():
        current_word = words[word_idx]
        displayed_word = current_word[:((i//2) % (len(current_word) + 1))]
        frame = next(spinner)
        
        # Professional gradient using reliability-themed colors
        gradient_text = f"""<div style="
            background: linear-gradient(
                45deg,
                #2C3E50,  /* Dark blue */
                #3498DB,  /* Professional blue */
                #2980B9   /* Trustworthy blue */
            );
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 20px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            padding: 20px;
            text-align: center;
            ">
            <style>
                @keyframes gradient {{
                    0% {{ background-position: 0% 50%; }}
                    100% {{ background-position: 100% 50%; }}
                }}
            </style>
            <div style="
                border: 2px solid #3498DB;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                box-shadow: 0 0 10px rgba(52, 152, 219, 0.2);
                ">
                {displayed_word} {frame}
            </div>
        </div>"""
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        
        i += 1
        if i % (len(current_word) * 2 + 15) == 0:
            word_idx = (word_idx + 1) % len(words)
            i = 0
            
        time.sleep(0.1)
