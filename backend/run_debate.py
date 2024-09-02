import agentsdebate

if __name__ == '__main__':
    ID=1
    prompt="Goverment policy regarding drug addiction problem"
    speaker_1="Donald Trump"
    speaker_2="Kamala Harris"
    agentsdebate.start_debate(ID, prompt, speaker_1,speaker_2)