import os
import spacy
from transformers import pipeline

# Suppress symlink warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load spaCy model for basic NLP processing
nlp = spacy.load("en_core_web_md")

# Load Hugging Face transformer model for advanced NLP tasks with explicit parameter
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True}  # Explicitly set the parameter
)

# Define tax laws and deductions
tax_laws = {
    "Section 80C": {
        "description": "Deductions on investments like PPF, EPF, ELSS, etc.",
        "limit": 150000,
        "investments": [
            {"name": "PPF", "description": "Public Provident Fund", "benefit": "Tax-free interest",
             "application_steps": [
                 "Visit a post office or bank where PPF is offered.",
                 "Fill out the PPF account opening form.",
                 "Submit KYC documents and initial deposit."
             ],
             "application_link": "https://www.nsiindia.gov.in/"},
            {"name": "ELSS", "description": "Equity Linked Savings Scheme", "benefit": "Tax-free returns after 3 years",
             "application_steps": [
                 "Choose a mutual fund company offering ELSS.",
                 "Complete the KYC process with the mutual fund.",
                 "Invest in ELSS through the company’s website or physical branch."
             ],
             "application_link": "https://groww.in/mutual-funds/equity-funds/elss-funds"},
            {"name": "NPS", "description": "National Pension Scheme",
             "benefit": "Additional deduction of Rs 50,000 under section 80CCD(1B)",
             "application_steps": [
                 "Visit the official NPS website or a designated Point of Presence (POP).",
                 "Complete the registration form and KYC process.",
                 "Make your contribution through online or offline modes."
             ],
             "application_link": "https://www.npscra.nsdl.co.in/"},
        ]
    },
    "Section 80D": {
        "description": "Deductions on health insurance premiums",
        "limit": 25000,
        "additional_for_senior_citizens": 50000,
        "investments": [
            {"name": "Health Insurance", "description": "Premium paid for health insurance",
             "benefit": "Deduction based on age group",
             "application_steps": [
                 "Purchase a health insurance policy from an insurance company.",
                 "Ensure the policy is in the name of the insured person.",
                 "Keep the premium receipts for claiming deductions."
             ],
             "application_link": "https://www.policybazaar.com/health-insurance/"},
        ]
    },
    "Section 80E": {
        "description": "Deductions on interest paid on education loans",
        "limit": "No upper limit",
        "investments": [
            {"name": "Education Loan", "description": "Loan taken for higher education",
             "benefit": "Deduction on interest paid",
             "application_steps": [
                 "Apply for an education loan from a bank or financial institution.",
                 "Keep records of loan disbursement and interest payments.",
                 "Claim the deduction while filing your income tax return."
             ],
             "application_link": "https://www.bankbazaar.com/education-loan.html"},
        ]
    }
}

# Define eligibility criteria
eligibility_criteria = {
    "Section 80C": {
        "eligible_person": "Individual and HUF",
        "max_age": "No age limit",
        "income_limit": "No income limit"
    },
    "Section 80D": {
        "eligible_person": "Individual and HUF",
        "max_age": 60,
        "income_limit": "No income limit"
    },
    "Section 80E": {
        "eligible_person": "Individual (for self, spouse, children)",
        "max_age": "No age limit",
        "income_limit": "No income limit"
    }
}

def get_tax_saving_options(income, age):
    applicable_schemes = []
    for section, data in tax_laws.items():
        if section in eligibility_criteria:
            criteria = eligibility_criteria[section]
            max_age = criteria.get("max_age", float('inf'))

            if isinstance(max_age, int) or isinstance(max_age, float):
                if age <= max_age:
                    applicable_schemes.append({
                        "section": section,
                        "description": data["description"],
                        "limit": data["limit"],
                        "investments": data["investments"]
                    })
            else:
                applicable_schemes.append({
                    "section": section,
                    "description": data["description"],
                    "limit": data["limit"],
                    "investments": data["investments"]
                })

    return applicable_schemes

def categorize_income(income):
    if income <= 250000:
        bracket = "No Tax"
    elif 250001 <= income <= 500000:
        bracket = "5% Tax Bracket"
    elif 500001 <= income <= 1000000:
        bracket = "20% Tax Bracket"
    else:
        bracket = "30% Tax Bracket"
    return bracket


def chat_bot():
    print("Hello! I'm your Tax Saving Assistant. How can I assist you today?")

    while True:
        user_input = input("You: ").strip().lower()
        doc = nlp(user_input)

        if any(token.lemma_ in ['hello', 'hi', 'hey'] for token in doc):
            print("Bot: Hi there! I can help you with tax-saving schemes. Would you like to know about them?")

        elif any(token.lemma_ in ['yes', 'yeah', 'yup'] for token in doc):
            try:
                income = float(input("Bot: Great! Can you please provide your annual income (in INR)? "))
                age = int(input("Bot: And your age? "))

                tax_bracket = categorize_income(income)
                print(f"Bot: Based on your income of INR {income}, you fall under the '{tax_bracket}'.")

                tax_saving_options = get_tax_saving_options(income, age)

                if tax_saving_options:
                    print("Bot: Here are some tax-saving schemes that you can consider:")
                    for idx, option in enumerate(tax_saving_options, start=1):
                        print(f"{idx}. Section: {option['section']}")
                        print(f"   Description: {option['description']}")
                        print(f"   Investment Limit: INR {option['limit']}")

                    scheme_choice = input(
                        "Bot: Would you like to explore details of any scheme? (Enter the number or 'no' to exit): ").strip().lower()

                    if scheme_choice != 'no':
                        try:
                            choice_idx = int(scheme_choice) - 1
                            if 0 <= choice_idx < len(tax_saving_options):
                                selected_option = tax_saving_options[choice_idx]
                                print(f"\nBot: Detailed Information about {selected_option['section']}:")
                                print(f"Description: {selected_option['description']}")
                                print(f"Investment Limit: INR {selected_option['limit']}")
                                for investment in selected_option['investments']:
                                    print(f"\nInvestment: {investment['name']}")
                                    print(f"Description: {investment['description']}")
                                    print(f"Benefit: {investment['benefit']}")
                                    print("Application Steps:")
                                    for step in investment['application_steps']:
                                        print(f"- {step}")
                                    print(f"Application Link: {investment['application_link']}")
                            else:
                                print("Bot: Invalid choice. Please try again.")
                        except ValueError:
                            print("Bot: Invalid input. Please enter a valid number.")
                else:
                    print("Bot: No applicable tax-saving schemes found based on the provided information.")

            except ValueError:
                print("Bot: Invalid input. Please enter numeric values for income and age.")

        elif any(token.lemma_ in ['no', 'nope', 'nah'] for token in doc):
            print("Bot: Alright! If you need any assistance, feel free to ask.")
            break

        elif any(token.lemma_ in ['exit', 'quit'] for token in doc):
            print("Bot: Goodbye! Have a great day!")
            break

        elif any(token.lemma_ in ['more', 'details', 'information'] for token in doc):
            # Use transformer model to answer questions about tax saving schemes
            context = " ".join([sent.text for sent in doc.sents])
            response = qa_pipeline(question="Tell me more about tax-saving schemes?", context=context)
            print(f"Bot: {response['answer']}")

        elif any(token.lemma_ in ['help', 'assistance'] for token in doc):
            print("Bot: I can assist you with information on tax-saving schemes. Let me know what you need help with!")

        else:
            print("Bot: I'm not sure how to respond to that. Can you please rephrase or ask something else?")

if __name__ == "__main__":
    chat_bot()