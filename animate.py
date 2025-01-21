def super_fancy_loading_animation():
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    phrases = [
        "Analyzing market insights",
        "Optimizing solutions",
        "Processing business intelligence",
        "Generating recommendations",
        "Evaluating opportunities",
        "Synthesizing data points",
        "Calculating outcomes",
        "Enhancing business value",
        "Streamlining workflows",
        "Maximizing efficiency"
    ]
    
    spinner = cycle(frames)
    phrase_index = 0
    fade_counter = 0
    typing_index = 0
    opacity = 0  # Start with 0 opacity for fade-in
    fade_in_speed = 0.1  # Controls how quickly text fades in
    max_dots = 3
    
    while not future.done():
        frame = next(spinner)
        current_phrase = phrases[phrase_index]
        
        # Handle opacity for fade-in effect
        if opacity < 1:
            opacity = min(1, opacity + fade_in_speed)
        
        # Get current display text with smooth typing effect
        if typing_index <= len(current_phrase):
            displayed_text = current_phrase[:typing_index]
            dots = ""
        else:
            displayed_text = current_phrase
            dots = "." * (((typing_index - len(current_phrase)) // 5) % (max_dots + 1))
        
        # Smoother typing speed with easing
        if typing_index < len(current_phrase):
            typing_speed = max(1, int((len(current_phrase) - typing_index) / 5))
            typing_index += typing_speed
        else:
            typing_index += 1
        
        # Handle phrase transition with fade-out
        if fade_counter >= 20:  # 2 seconds display time
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            typing_index = 0
            opacity = 0  # Reset opacity for next phrase
        
        fade_counter += 1
        
        gradient_text = f"""
        <div style="
            font-size: 22px;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-weight: 400;
            letter-spacing: -0.2px;
            padding: 3px 0;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 12px;
            opacity: {opacity};
            transition: opacity 0.3s ease-in-out;
            ">
            <span style="
                color: rgb(45, 55, 72);
                text-rendering: optimizeLegibility;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                ">{displayed_text}{dots}</span>
            <span style="
                color: rgb(255, 75, 75);
                font-size: 20px;
                transform: translateY(-1px);
                animation: pulse 1s infinite ease-in-out;
                ">{frame}</span>
        </div>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');
            @keyframes pulse {
                0% { opacity: 0.7; }
                50% { opacity: 1; }
                100% { opacity: 0.7; }
            }
        </style>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.05)
