import re
import ollama
import spacy
from pydantic import BaseModel, validator
import pandas as pd

# ---------------------------
# Pydantic Model for Validation
# ---------------------------
class CleanedLine(BaseModel):
    line: str

    @validator("line")
    def non_empty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Line cannot be empty")
        return value

# ---------------------------
# spaCy-based Pre-filtering Function
# ---------------------------
def spacy_prefilter(text: str) -> str:
    """
    Pre-filter the text using spaCy and regex by:
      - Removing tags enclosed in square brackets (e.g., [HEADING], [START], etc.)
      - Splitting the text into sentences
      - Removing duplicate or tag-only sentences (handled case-insensitively)
      - Joining sentences into a single cleaned paragraph
    """
    # Remove any content within square brackets (these are tags)
    text = re.sub(r"\[.*?\]", "", text)
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Extract sentences and clean whitespace
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Define a set of isolated tag words (if they appear as entire sentences)
    tag_words = {"start", "heading", "complementary", "another"}
    filtered_sentences = [sent for sent in sentences if sent.lower() not in tag_words]
    
    # Remove duplicate sentences (case-insensitive)
    unique_sentences = []
    seen = set()
    for sent in filtered_sentences:
        low = sent.lower()
        if low not in seen:
            seen.add(low)
            unique_sentences.append(sent)
    
    # Join sentences into one cleaned paragraph
    cleaned_text = " ".join(unique_sentences)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

# ---------------------------
# Function to Get Reformatted Output from phi4:14b-fp16 Model via Ollama
# ---------------------------
def get_reformatted_output(text: str) -> str:
    """
    Instruct the LLM only to reformat the pre-filtered text. The prompt
    instructs it to simply join broken lines and fix minor formatting issues,
    without removing or altering any of the content.
    """
    # Pre-filter the text first
    filtered_text = spacy_prefilter(text)
    
    # Build a prompt that instructs the model to reformat the text without removal
    prompt = (
        "Please reformat the following text by correcting minor formatting issues "
        "such as broken lines or misplaced punctuation. Do not remove or alter any content; "
        "simply return the text with improved formatting. Return the text exactly as is "
        "with only formatting corrections:\n\n"
        + filtered_text
    )
    
    # Call the phi4:14b-fp16 model via Ollama
    result = ollama.generate(model="phi4:14b-fp16", prompt=prompt)
    return result["response"]

# ---------------------------
# Function to Validate and Filter Lines with Pydantic
# ---------------------------
def filter_and_validate_lines(text: str) -> list:
    """
    Split the reformatted text into individual lines, validate each using Pydantic,
    and return a list of valid dictionary entries.
    """
    raw_lines = text.splitlines()
    valid_entries = []
    for line in raw_lines:
        if line.strip():
            try:
                entry = CleanedLine(line=line)
                valid_entries.append(entry.dict())
            except Exception as e:
                print(f"Skipping line: {line}\nError: {e}")
    return valid_entries

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    raw_text = """
    [START: COMPLEMENTARY USER ENTITY CONTROLS (CUECS)]
    Oracle IT general controls were designed with the assumption that certain controls would be implemented
    by user entities (or “customers”). This section describes additional controls that customers must have in
    operation to complement the controls of Oracle. The list of customer control considerations presented below
    and those presented with certain specified trust services criteria do not represent a comprehensive set of
    all the controls that should be employed by customers. Customers may be required to implement additional
    [HEADING]: administrative or technical controls to meet their business and legal needs.
    [HEADING]: Applicable Trust Services Criteria
    [ANOTHER COMPLEMENTARY HEADING]: Complementary User Entity Controls (CUECs)
    [HEADING]: CC1.3 – COSO Principle 3: Management
    [HEADING]: establishes, with board oversight,
    [HEADING]: structures, reporting lines, and appropriate
    [HEADING]: authorities and responsibilities in the pursuit
    [HEADING]: of objectives.
    [HEADING]: The customer is responsible for defining its
    [HEADING]: organizational structures, reporting lines, authorities,
    [HEADING]: and responsibilities for the design, development,
    [HEADING]: implementation, operation, maintenance and
    [HEADING]: monitoring of its application in accordance with
    [HEADING]: applicable laws, regulations standards in order to
    [HEADING]: meet its commitments and requirements as they
    [HEADING]: relate to security, availability, and confidentiality.
    [HEADING]: CC2.1 – COSO Principle 13: The entity
    [HEADING]: obtains or generates and uses relevant,
    [HEADING]: quality information to support the
    [HEADING]: functioning of internal control.
    [HEADING]: The customer is responsible for ensuring the
    [HEADING]: completeness and accuracy of the financial data.
    [HEADING]: The customer is responsible for application
    [HEADING]: functionality, configuration, and transaction
    [HEADING]: processing.
    [HEADING]: The customer is responsible for determining their
    [HEADING]: business processes and defining, or working with the
    [HEADING]: implementer to define, any configuration controls
    [HEADING]: required to support the business processes.
    [HEADING]: CC2.2 – COSO Principle 14: The entity
    [HEADING]: internally communicates information,
    [HEADING]: including objectives and responsibilities for
    [HEADING]: internal control, necessary to support the
    [HEADING]: functioning of internal control.
    [HEADING]: The customer is responsible for communicating
    [HEADING]: confidentiality commitments and requirements to
    [HEADING]: internal and external users of its application.
    [HEADING]: CC2.3 – COSO Principle 15: The entity
    [HEADING]: communicates with external parties
    [HEADING]: regarding matters affecting the functioning
    [HEADING]: of internal control.
    [HEADING]: The customer is responsible for notifying Oracle of
    [HEADING]: any unauthorized use of, and other known or
    [HEADING]: suspected breach of, security, including
    [HEADING]: compromised user accounts.
    [HEADING]: The customer is responsible for reviewing incident
    [HEADING]: response details provided by Oracle and to initiate
    [HEADING]: inquiry or follow-up as appropriate.
    [HEADING]: The customer is responsible for communicating
    [HEADING]: confidentiality commitments and requirements to
    [HEADING]: internal and external users of its application.
    [HEADING]: The customer is responsible for submitting incident
    [HEADING]: tickets (i.e. data restores) through the My Oracle
    [HEADING]: Support (MOS) customer portal. Customer requests
    [HEADING]: are recorded and tracked within an internal ticketing
    [HEADING]: system through resolution. The ticketing system is
    [HEADING]: utilized to document, prioritize, escalate, and resolve
    [HEADING]: problems affecting contracted services. Customer
    [HEADING]: requests are managed according to established
    [HEADING]: service level agreements.
    [HEADING]: ORACLE CONFIDENTIAL
    [HEADING]: 31 of 97
    [HEADING]: Applicable Trust Services Criteria
    [ANOTHER COMPLEMENTARY HEADING]: Complementary User Entity Controls (CUECs)
    [HEADING]: CC6.1 – The entity implements logical
    [HEADING]: access security software, infrastructure, and
    [HEADING]: architectures over protected information
    [HEADING]: assets to protect them from security events
    [HEADING]: to meet the entity’s objectives.
    [HEADING]: The customer is responsible for all aspects of
    [HEADING]: security relevant to its application.
    [HEADING]: The customer is responsible for designing,
    [HEADING]: developing, testing, implementing, operating and
    [HEADING]: maintaining administrative and technical safeguards
    [HEADING]: to prevent or detect unauthorized access, use, and
    [HEADING]: disclosure during input, processing, retention, output,
    [HEADING]: and disposition of data to, in or from its application.
    [HEADING]: The customer is responsible for confidentiality of
    [HEADING]: passwords and user IDs assigned by them.
    [HEADING]: The customer is responsible for ensuring the
    [HEADING]: confidentiality of any user accounts and passwords
    [HEADING]: assigned to them for use with Oracle’s systems.
    [HEADING]: The customer is responsible for managing
    [HEADING]: application-level access for their employees
    [HEADING]: throughout an employee’s relationship with the entity
    [HEADING]: (e.g., onboarding, termination, role changes, etc.).
    [HEADING]: CC6.2 – Prior to issuing system credentials
    [HEADING]: and granting system access, the entity
    [HEADING]: registers and authorizes new internal and
    [HEADING]: external users whose access is
    [HEADING]: administered by the entity. For those users
    [HEADING]: whose access is administered by the entity,
    [HEADING]: user system credentials are removed when
    [HEADING]: user access is no longer authorized.
    [HEADING]: The customer is responsible for administering
    [HEADING]: application security rights.
    [HEADING]: The customer is responsible for removing terminated
    [HEADING]: employees’ access.
    [HEADING]: The customer is responsible for defining authorized
    [HEADING]: application administrators within its application,
    [HEADING]: ensuring that these privileges are restricted to
    [HEADING]: authorized individuals, and for periodically reviewing
    [HEADING]: the security configurations and access rights for
    [HEADING]: appropriateness.
    [HEADING]: The customer is responsible for managing
    [HEADING]: application-level access for their employees
    [HEADING]: throughout an employee’s relationship with the entity
    [HEADING]: (e.g., onboarding, termination, role changes, etc.).
    [HEADING]: ORACLE CONFIDENTIAL
    [HEADING]: 32 of 97
    [HEADING]: Applicable Trust Services Criteria
    [ANOTHER COMPLEMENTARY HEADING]: Complementary User Entity Controls (CUECs)
    [HEADING]: CC6.6 – The entity implements logical
    [HEADING]: access security measures to protect against
    [HEADING]: threats from sources outside its system
    [HEADING]: boundaries.
    [HEADING]: The customer is responsible for configuring
    [HEADING]: application password parameters and complexity
    [HEADING]: requirements.
    [HEADING]: The customer is responsible for confidentiality of
    [HEADING]: passwords and user IDs assigned by them.
    [HEADING]: The customer is responsible for administering
    [HEADING]: application security rights.
    [HEADING]: The customer is responsible for removing terminated
    [HEADING]: employees’ access.
    [HEADING]: The customer is responsible for defining authorized
    [HEADING]: application administrators within its application,
    [HEADING]: ensuring that these privileges are restricted to
    [HEADING]: authorized individuals, and for periodically reviewing
    [HEADING]: the security configurations and access rights for
    [HEADING]: appropriateness.
    [HEADING]: The customer is responsible for defining authorized
    [HEADING]: application administrators within the application, and
    [HEADING]: for periodically reviewing the access rights for all end
    [HEADING]: users are appropriate.
    [HEADING]: The customer is responsible for immediately notifying
    [HEADING]: Oracle of any actual or suspected information
    [HEADING]: security breaches, including compromised user
    [HEADING]: accounts.
    [HEADING]: CC8.1 – The entity authorizes, designs,
    [HEADING]: develops or acquires, configures,
    [HEADING]: documents, tests, approves, and
    [HEADING]: implements changes to infrastructure, data,
    [HEADING]: software, and procedures to meet its
    [HEADING]: objectives.
    [HEADING]: The customer is responsible for any customized
    [HEADING]: changes made to its environment including, but not
    [HEADING]: limited to, virtual networks, operating systems, virtual
    [HEADING]: machines, databases, storage, and applications.
    [HEADING]: The customer is responsible for reviewing release
    [HEADING]: notes and other notices of changes and to evaluate,
    [HEADING]: and if necessary, take steps to mitigate, the effects of
    [HEADING]: any changes.
    [HEADING]: The customer is responsible for ensuring sufficient
    [HEADING]: controls for the implementation of the application.
    [HEADING]: The customer is responsible for ensuring the
    [HEADING]: integration with systems external to the application.
    [HEADING]: The customer is responsible for the change
    [HEADING]: management process regarding specific
    [HEADING]: customization(s).
    [HEADING]: The customer is responsible for data changes (e.g.
    [HEADING]: update, insertion and deletion of data).
    [HEADING]: The customer is responsible for the configuration of
    [HEADING]: their application and any third-party applications.
    [HEADING]: ORACLE CONFIDENTIAL
    [HEADING]: 33 of 97
    [HEADING]: Applicable Trust Services Criteria
    [ANOTHER COMPLEMENTARY HEADING]: Complementary User Entity Controls (CUECs)
    [HEADING]: C1.1 – The entity identifies and maintains
    [HEADING]: confidential information to meet the entity’s
    [HEADING]: objectives related to confidentiality.
    [HEADING]: The customer is responsible for ensuring that they do
    [HEADING]: not enter confidential data in support tickets and
    [HEADING]: requests.
    [HEADING]: The customer is responsible for designing,
    [HEADING]: developing, testing, implementing, operating, and
    [HEADING]: maintaining administrative and technical safeguards
    [HEADING]: to prevent or detect unauthorized access, use, and
    [HEADING]: disclosure during input, processing, retention, output,
    [HEADING]: and disposition of data to, in or from its application.
    [HEADING]: The customer is responsible for communicating
    [HEADING]: confidentiality commitments and requirements to
    [HEADING]: internal and external users of its application.
    [HEADING]: The customer is responsible for confidentiality of
    [HEADING]: passwords and user IDs assigned by them.
    [HEADING]: C1.2 - The entity disposes of confidential
    [HEADING]: information to meet the entity’s objectives
    [HEADING]: related to confidentiality.
    [HEADING]: The customer is responsible for notifying Oracle of
    [HEADING]: their intent to discontinue the use of the Service after
    [HEADING]: the end of the Service Period.
    [HEADING]: ORACLE CONFIDENTIAL
    [HEADING]: 34 of 97
    [ANOTHER COMPLEMENTARY HEADING]: COMPLEMENTARY SUBSERVICE ORGANIZATION CONTROLS
    [HEADING]: (CSOCS)
    Oracle uses subservice organizations to provide data center hosting services. These data center facilities
    include controls over physical security supporting the System. This description includes only those controls
    [HEADING]: expected at the subservice organizations and does not include controls at Oracle.
    [HEADING]: Trust Services
    [HEADING]: Criteria
    [ANOTHER COMPLEMENTARY HEADING]: Complementary Subservice Organization Controls
    [HEADING]: Common Criteria 2.3
    [HEADING]: The subservice organization is responsible for informing Oracle of all
    [HEADING]: physical security breaches, failures, identified vulnerabilities, and incidents.
    [HEADING]: Common Criteria 6.4
    [HEADING]: The subservice organization is responsible for restricting physical access to
    [HEADING]: production systems.
    [HEADING]: Common Criteria 6.4
    [HEADING]: The subservice organization is responsible for restricting physical access to
    [HEADING]: offline storage and backup media to help ensure that application and data
    [HEADING]: files are securely stored.
    [HEADING]: Availability Criteria 1.2
    [HEADING]: The subservice organization is responsible for equipping the data center
    [HEADING]: facilities with environmental security safeguards and utilizing an
    [HEADING]: environmental monitoring application to monitor for environmental events to
    [HEADING]: help ensure that systems are maintained in a manner that helps ensure
    [HEADING]: system availability.
    [HEADING]: ORACLE CONFIDENTIAL
    [HEADING]: 35 of 97
    """
    
    # Get the reformatted output from the phi4:14b-fp16 model
    reformatted_output = get_reformatted_output(raw_text)
    print("Reformatted Output:\n")
    print(reformatted_output)
    
    # Save the reformatted output to a text file
    with open("reformatted_output.txt", "w", encoding="utf-8") as f:
        f.write(reformatted_output)
    
    # Validate and filter the output lines using Pydantic
    validated_lines = filter_and_validate_lines(reformatted_output)
    
    # Create a Pandas DataFrame from the validated lines
    df = pd.DataFrame(validated_lines)
    
    print("\nDataFrame with Validated Lines:")
    print(df)
    
    # Save the DataFrame to a CSV file
    df.to_csv("reformatted_output.csv", index=False)
