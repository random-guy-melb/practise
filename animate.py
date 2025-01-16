def super_fancy_loading_animation():
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    phrases = [
        "Analyzing market insights...",
        "Optimizing solutions...",
        "Processing business intelligence...",
        "Generating strategic recommendations...",
        "Evaluating opportunities...",
        "Synthesizing data points...",
        "Calculating optimal outcomes...",
        "Enhancing business value...",
        "Streamlining workflows...",
        "Maximizing efficiency..."
    ]
    
    spinner = cycle(frames)
    phrase_index = 0
    fade_counter = 0
    is_fading = False
    
    while not future.done():
        frame = next(spinner)
        current_phrase = phrases[phrase_index]
        
        # Handle fading animation
        if fade_counter >= 30:  # 3 seconds (30 * 0.1s)
            is_fading = True
            opacity = max(0, 1 - ((fade_counter - 30) / 5))  # Fade out over 0.5s
        else:
            is_fading = False
            opacity = 1
            
        # Update counters and phrase
        fade_counter += 1
        if fade_counter >= 35:  # 3.5 seconds total
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            
        # Create the HTML with fade effect
        gradient_text = f"""
        <div style="
            font-size: 24px;
            font-family: sans-serif;
            padding: 3px 0;
            text-align: center;
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            ">
            <div style="
                opacity: {opacity};
                transition: opacity 0.5s ease;
                color: rgb(55, 65, 81);
                font-weight: 500;
                margin-bottom: 16px;
                ">{current_phrase}</div>
            <div style="display: flex; gap: 8px; justify-content: center;">
                <span style="
                    width: 8px;
                    height: 8px;
                    background-color: rgb(59, 130, 246);
                    border-radius: 50%;
                    animation: pulse 1s infinite;
                    animation-delay: 0s;
                    "></span>
                <span style="
                    width: 8px;
                    height: 8px;
                    background-color: rgb(59, 130, 246);
                    border-radius: 50%;
                    animation: pulse 1s infinite;
                    animation-delay: 0.2s;
                    "></span>
                <span style="
                    width: 8px;
                    height: 8px;
                    background-color: rgb(59, 130, 246);
                    border-radius: 50%;
                    animation: pulse 1s infinite;
                    animation-delay: 0.4s;
                    "></span>
            </div>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(0.8); opacity: 0.5; }
                50% { transform: scale(1.2); opacity: 1; }
                100% { transform: scale(0.8); opacity: 0.5; }
            }
        </style>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.1)
