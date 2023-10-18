from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

if __name__ == "__main__":
    print("Hello langChain!")

    information = """Amelia Mary Earhart (/ˈɛərhɑːrt/ AIR-hart, born July 24, 1897; disappeared July 2, 1937; declared dead January 5, 1939) was an American aviation pioneer and writer.[2][Note 1] Earhart was the first female aviator to fly solo across the Atlantic Ocean.[4] She set many other records,[3][Note 2] was one of the first aviators to promote commercial air travel, wrote best-selling books about her flying experiences, and was instrumental in the formation of The Ninety-Nines, an organization for female pilots.[6]
          Born and raised in Atchison, Kansas, and later in Des Moines, Iowa, Earhart developed a passion for adventure at a young age, steadily gaining flying experience from her twenties. In 1928, Earhart became the first female passenger to cross the Atlantic by airplane (accompanying pilot Wilmer Stultz), for which she achieved celebrity status. In 1932, piloting a Lockheed Vega 5B, Earhart made a nonstop solo transatlantic flight, becoming the first woman to achieve such a feat. She received the United States Distinguished Flying Cross for this accomplishment.[7] In 1935, Earhart became a visiting faculty member at Purdue University as an advisor to aeronautical engineering and a career counselor to female students. She was also a member of the National Woman's Party and an early supporter of the Equal Rights Amendment.[8][9] Known as one of the most inspirational American figures in aviation from the late 1920s throughout the 1930s, Earhart's legacy is often compared to the early aeronautical career of pioneer aviator Charles Lindbergh, as well as to figures like First Lady Eleanor Roosevelt for their close friendship and lasting impact on the issue of women's causes from that period.
          """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))
