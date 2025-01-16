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
    
    while not future.done():
        frame = next(spinner)
        current_phrase = phrases[phrase_index]
        
        # Handle typing effect
        if typing_index < len(current_phrase):
            typing_index += 1
        
        # Handle phrase transition
        if fade_counter >= 15:  # Reduced to 1.5 seconds display time
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            typing_index = 0  # Reset typing index for new phrase
            
        fade_counter += 1
        
        # Get currently visible text
        visible_text = current_phrase[:typing_index]
        
        gradient_text = f"""
        <div style="
            font-size: 24px;
            font-family: monospace;
            padding: 3px 0;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 4px;
            color: rgb(55, 65, 81);
            ">
            <span>{visible_text}</span>
            <span style="
                display: inline-flex;
                gap: 4px;
                margin-left: 4px;
                ">
                <span style="opacity: 0.7">.</span>
                <span style="opacity: 0.8">.</span>
                <span style="opacity: 0.9">.</span>
            </span>
        </div>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.05)  # Reduced sleep time for faster typing
