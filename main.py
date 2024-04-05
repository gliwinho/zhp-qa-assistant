from core.chain import question_chain

if __name__ == "__main__":
    while True:
        content = input('>> ')
        output = question_chain.invoke({
            "input": content
        })
        print(output["text"])
