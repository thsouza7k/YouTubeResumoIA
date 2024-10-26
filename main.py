from pytubefix import YouTube
import os
from groq import Groq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# API key
api_key = os.getenv("GROQ_API_KEY")

#Extract audio from video
video_url = "https://www.youtube.com/watch?v=EddPAllx3sU"
yt= YouTube(video_url)
base_dir = f"audios"
stream = yt.streams.get_audio_only()
stream.download(mp3 = True, filename = f"{yt.title}", output_path = f"{base_dir}")
print(f"Downloading {yt.title}.")


# Initialize the Groq client
client = Groq(api_key=api_key)


#Transcribe audio
audio_file = "audios/"+yt.title+".mp3"
# Open the audio file
with open(audio_file, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=(audio_file, file.read()), # Required audio file
      model="whisper-large-v3", # Required model to use for transcription: "whisper-large-v3"/distil-whisper-large-v3-en para vídeos pt-br
      #language = "pt"
      temperature=0.0,
    )
 
result = transcription.text

print("Transcrição concluida")

# .txt file transcription
#with open(f'{yt.title}.txt', 'w', encoding='utf-8') as f:
#    f.write(result)

#summary template 

template= f"""
    Você é um assistente para resumir vídeos.

    Você receberá o conteúdo transcrito de um vídeo entre <t></t> e deverá
    criar um resumo didatico e rico em detalhes, 
    em MARKDOWN, contendo: 

   

    1. introdução
        - Escreva brevemente uma introdução sobre o que foi falado no vídeo. 
        - [{yt.title}]({video_url})
    2. Os principais pontos abordados, em ordem cronológica. Numere-os.
        - Escreva um texto bem completo sobre cada um dos pontos. Seja específico e detalhista.
    
    3. Uma conclusão
        - Escreva uma curta conclusão sobre o vídeo
    
    <t>{result}</t>
    """


response = client.chat.completions.create(

    #

    # Required parameters

    #

    messages=[

        # Set an optional system message. This sets the behavior of the

        # assistant and can be used to provide specific instructions for

        # how it should behave throughout the conversation.


        {

            "role": "user",

            "content": template,

        }

    ],


    # The language model which will generate the completion.

    model="llama-3.1-70b-versatile", #llama-3.2-3b-preview

    #

    # Optional parameters

    #


    # Controls randomness: lowering results in less random completions.

    # As the temperature approaches zero, the model will become deterministic

    # and repetitive.

    temperature=0.3,


    # The maximum number of tokens to generate. Requests can use up to

    # 32,768 tokens shared between prompt and completion.

    max_tokens=8000,


    # Controls diversity via nucleus sampling: 0.5 means half of all

    # likelihood-weighted options are considered.

    top_p=1,


    # A stop sequence is a predefined or user-specified text string that

    # signals an AI to stop generating content, ensuring its responses

    # remain focused and concise. Examples include punctuation marks and

    # markers like "[end]".

    stop=None,


    # If set, partial message deltas will be sent.

    stream=False,

)


summary = response.choices[0].message.content

pasta = 'summaries/'

if not os.path.exists(pasta):
    os.makedirs(pasta)

with open(f'{pasta}{yt.title}.md', 'w', encoding='utf-8') as f:
    f.write(summary)

print("Resumo concluído!")

