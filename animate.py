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
    i = 0
    
    while not future.done():
        frame = next(spinner)
        current_phrase = phrases[phrase_index]
        
        # Get current display text
        if typing_index <= len(current_phrase):
            displayed_text = current_phrase[:typing_index]
            dots = ""
        else:
            displayed_text = current_phrase
            dots = "." * min(typing_index - len(current_phrase), max_dots)
        
        # Increment typing index
        if typing_index < len(current_phrase) + max_dots:
            typing_index += 1
        
        # Handle phrase transition
        if fade_counter >= 15:  # 1.5 seconds display time
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            typing_index = 0  # Reset typing index for new phrase
        
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
                ">{frame}</span>
        </div>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');
        </style>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.05)



"""
 /* Container for the entire chat interface */
        .stChatFloatingInputContainer {
            padding-bottom: 20px;
        }
        
        /* Message container styling */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        
        /* User message specific styling */
        .user-message {
            background-color: #2b313e;
            border-radius: 15px;
            padding: 10px 15px;
            margin-left: auto;  /* Push message to the right */
            margin-right: 0;
            max-width: fit-content;  /* Adjust width to content */
            text-align: right;
        }
        
        /* Assistant message specific styling */
        .assistant-message {
            background-color: #343541;
            border-radius: 15px;
            padding: 10px 15px;
            margin-right: auto;  /* Push message to the left */
            margin-left: 0;
            max-width: 80%;  /* Limit width for readability */
        }
"""
