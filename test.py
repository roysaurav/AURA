import streamlit as st
import pandas as pd
import time

# --- LangChain & Google Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


# --- Configuration ---
# Check for API Key in Streamlit Secrets
if "google_api_key" not in st.secrets:
    st.error("🔴 Google API Key not found. Please add it to your Streamlit Secrets.")
    st.stop()
    
CANADIAN_MARGINAL_TAX_RATE = 0.5353  # Top marginal rate in Ontario
CANADIAN_CAPITAL_GAINS_INCLUSION_RATE = 0.50

# --- Helper Functions ---
def load_data():
    """Loads portfolio data from CSV."""
    try:
        df = pd.read_csv("portfolio_data.csv")
        df['market_value'] = df['shares'] * df['current_price']
        df['cost_basis'] = df['shares'] * df['avg_cost_price']
        df['unrealized_gain_loss'] = df['market_value'] - df['cost_basis']
        return df
    except FileNotFoundError:
        st.error("Error: `portfolio_data.csv` not found. Please make sure it's in the same directory.")
        return None

def format_currency(amount):
    """Formats a number as CAD currency."""
    return f"${amount:,.2f}"

# --- Agent Logic ---
class AuraAgent:
    def __init__(self, portfolio_df, client_id):
        self.portfolio = portfolio_df[portfolio_df['client_id'] == client_id].copy()
        self.client_name = self.portfolio['client_name'].iloc[0]
        self.client_persona = self.portfolio['client_persona'].iloc[0]
        self.log = []

    def find_tax_loss_harvest_opportunity(self, min_loss_threshold=5000):
        """
        Simulates the agent finding the best TLH opportunity.
        This is a simplified "optimization" for the POC.
        """
        self.log.append(f"✅ INITIATING... Monitoring {self.client_name}'s portfolio.")
        time.sleep(1)
        
        candidates = self.portfolio[self.portfolio['unrealized_gain_loss'] < 0].sort_values(by='unrealized_gain_loss')
        
        if candidates.empty:
            self.log.append("ℹ️ CONCLUSION... No significant unrealized losses found.")
            return None

        best_opportunity = candidates.iloc[0]
        loss_amount = abs(best_opportunity['unrealized_gain_loss'])
        self.log.append(f"🔍 SCANNING... Identified unrealized loss in {best_opportunity['ticker']}: {format_currency(best_opportunity['unrealized_gain_loss'])}.")
        time.sleep(1)

        if loss_amount < min_loss_threshold:
            self.log.append(f"ℹ️ CONCLUSION... Loss of {format_currency(loss_amount)} is below the ${min_loss_threshold:,.0f} threshold.")
            return None
        
        tax_alpha = loss_amount * CANADIAN_CAPITAL_GAINS_INCLUSION_RATE * CANADIAN_MARGINAL_TAX_RATE
        self.log.append(f"🧠 EVALUATING... Potential tax alpha is {format_currency(tax_alpha)}.")
        time.sleep(1)

        self.log.append("🛡️ RULE CHECK... Action does not violate 30-day superficial loss rule.")
        time.sleep(0.5)
        
        self.log.append("🎯 CONCLUSION... Opportunity meets criteria. Generating proposal.")
        return {
            "ticker": best_opportunity['ticker'],
            "shares": best_opportunity['shares'],
            "loss_amount": loss_amount,
            "tax_alpha": tax_alpha
        }

    def generate_client_communication(self, proposal):
        """Generates the client-facing narrative using LangChain and Google Gemini."""
        
        # 1. Instantiate the LLM using the API key from st.secrets
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets["google_api_key"],
            temperature=0.4
        )
        
        # 2. Define the UPDATED, more concise Prompt Template
        prompt_template = PromptTemplate.from_template(
            """
            You are AURA, an AI assistant for a wealth advisor. Your goal is to be extremely concise and clear.
            Draft a short message to a client about a tax-loss harvesting trade.

            **Follow this exact format and tone:**
            "Hi {client_first_name},
            
            This week we took advantage of market movements to strategically improve your portfolio's tax efficiency... By realizing a loss of {realized_loss}... This is projected to lower your upcoming tax bill by an estimated {tax_alpha}..."
            
            **Do not add any extra sentences or pleasantries.**
            """
        )

        # 3. Create the LangChain Chain
        chain = prompt_template | llm | StrOutputParser()

        # 4. Invoke the chain with the required inputs
        try:
            response = chain.invoke({
                "client_first_name": self.client_name.split()[0],
                "realized_loss": format_currency(proposal['loss_amount']),
                "tax_alpha": format_currency(proposal['tax_alpha'])
            })
            return response
        except Exception as e:
            st.error(f"Failed to connect to Google Gemini API: {e}")
            return "Error: Could not generate communication."

# --- Streamlit UI (No changes needed here) ---
st.set_page_config(page_title="AURA")
st.title("🤖 AURA: Agentic Universal Reporting Assistant")
st.markdown(f"_(Powered by LangChain & Google Gemini)_")
st.markdown(f"*_Tuesday, July 29, 2025 - Toronto, Ontario_*")

st.header("Step 1: The Advisor's Directive")

portfolio_df = load_data()

if portfolio_df is not None:
    client_list = portfolio_df['client_name'].unique()
    selected_client_name = st.selectbox("Select a Client to Monitor:", client_list)
    
    selected_client_id = portfolio_df[portfolio_df['client_name'] == selected_client_name]['client_id'].iloc[0]

    with st.expander("View Selected Client's Portfolio"):
        st.dataframe(portfolio_df[portfolio_df['client_id'] == selected_client_id])

    if st.button("🚀 Engage AURA Agent", type="primary"):
        st.session_state.agent = AuraAgent(portfolio_df, selected_client_id)
        st.session_state.proposal = None
        st.session_state.communication = None
        
if 'agent' in st.session_state:
    st.divider()
    st.header("Step 2: The Agent's Live Analysis")
    
    agent = st.session_state.agent
    
    if 'proposal' not in st.session_state or st.session_state.proposal is None:
        with st.spinner("AURA is thinking..."):
            st.session_state.proposal = agent.find_tax_loss_harvest_opportunity()
            
            log_placeholder = st.empty()
            log_text = ""
            for log_entry in agent.log:
                log_text += log_entry + "\n"
                log_placeholder.code(log_text, language="log")
                time.sleep(0.5) 
    else:
        st.code('\n'.join(agent.log), language="log")

    st.divider()
    st.header("Step 3: The Agent's Proposal")
    
    proposal = st.session_state.proposal
    
    if proposal:
        col1, col2 = st.columns(2)
        col1.metric("Actionable Loss", format_currency(proposal['loss_amount']))
        col2.metric("Estimated Tax Alpha", format_currency(proposal['tax_alpha']))

        st.info(f"**Proposal:** Sell {proposal['shares']} shares of **{proposal['ticker']}** to realize the capital loss and generate tax alpha.")
        
        if st.button("✅ Approve & Generate Client Communication"):
             with st.spinner("AURA is drafting the communication..."):
                st.session_state.communication = agent.generate_client_communication(proposal)
    else:
        st.success("AURA found no actionable opportunities that meet the current criteria.")
        
if 'communication' in st.session_state and st.session_state.communication:
    st.divider()
    st.header("Step 4: AURA-Generated Client Communication")
    st.text(st.session_state.communication)