from __future__ import annotations

from typing import Any

_EXAMPLES = {
    "kb_qa": """Question: Does Sarah Chen have direct reports?
Answer: True

Question: How many employees are working on the Phoenix Project?
Answer: 7

Question: When did Alex Martinez start working at TechCorp?
Answer: 2023-03-15

Question: Who reports to Jennifer Kim?
Answer: Michael Torres, Lisa Wang, James Rodriguez

Question: What team is David Thompson on?
Answer: Engineering Platform Team""",
    "topic_qa": """Question: Was cloud migration discussed on 2024-05-15?
Answer: False

Question: How many documents mention API security?
Answer: 23

Question: When was database optimization first mentioned?
Answer: 2024-01-10

Question: Who has discussed machine learning infrastructure?
Answer: Emma Wilson, Robert Chen, Nina Patel

Question: What was the main topic of the document sent on 2024-06-20?
Answer: quarterly performance review""",
    "integrated_qa": """Question: Did anyone who reports to Carlos Rivera discuss budget planning before 2024-07-01?
Answer: True

Question: How many employees who work at DataCorp sent emails about customer retention?
Answer: 12

Question: When did members of the Marketing Team first mention product launch strategy?
Answer: 2024-02-28

Question: Which employees who work on the Atlas Project discussed system architecture with people outside their company?
Answer: Sophie Anderson, Marcus Lee

Question: What role did Katherine Brooks hold between 2023-01-01 and 2023-06-30?
Answer: Senior Product Manager""",
}

_ANSWER_INSTRUCTIONS = {
    "bool": "Answer EXCLUSIVELY with a boolean value (True or False).",
    "int": "Answer EXCLUSIVELY with an integer value.",
    "date": "Answer EXCLUSIVELY with a date in the format YYYY-MM-DD.",
    "List[str]": "Answer EXCLUSIVELY with a comma-separated list of strings.",
    "str": "Answer EXCLUSIVELY with a string.",
}

_DESCRIPTIONS = {
    "rag": "a set of documents of a company",
    "kb_qa": "a set of entities and relationships in a company",
    "topic_qa": "a set of documents of a company",
    "integrated_qa": (
        "a set of entities and relationships including corresponding documents "
        "of a company"
    ),
}

RAG_SYSTEM = """You are a document retrieval assistant specializing in company documentation search.

Use the retrieve tool to find relevant information.

BENCHMARK CONSTRAINTS:
- You are limited to {tool_limit} tool calls maximum. Use them wisely.
- Return results in the exact format specified by the user."""

_ENTITY_SCHEMA = """ENTITIES:

gold_company:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Company name
- "aliases" (TEXT[]): Company aliases/acronyms

gold_department:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Department name
- "aliases" (TEXT[]): Department aliases/acronyms

gold_team:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Team name
- "aliases" (TEXT[]): Team aliases/acronyms

gold_employee:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Employee full name
- "aliases" (TEXT[]): Employee aliases
- "title" (TEXT): Job title
- "personal_email" (TEXT): Personal email address
- "work_email" (TEXT): Work email address

gold_task:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Task name
- "aliases" (TEXT[]): Task aliases
- "description" (TEXT): Task description
- "status" (TEXT): Task status

gold_project:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "name" (TEXT): Project name
- "aliases" (TEXT[]): Project aliases

gold_meeting:
- "id" (VARCHAR): Unique identifier
- "type" (TEXT): Entity type
- "title" (TEXT): Meeting title
- "meeting_type" (TEXT): Type of meeting
- "topic" (TEXT): Meeting topic (nullable)
- "date" (DATE): Meeting date

RELATIONSHIPS:

IMPORTANT: ALL relationships between entities are stored in gold_relationship table.
Entity tables do NOT contain foreign keys - always JOIN with the relationship table.

gold_relationship:
- "id" (VARCHAR): Unique identifier
- "type" (VARCHAR): Relationship type - One of: worksAt, worksOn, memberOf, reportsTo,
  teamBelongsToDepartment, departmentBelongsToCompany, departmentOwnsProject, hasTask,
  organize, attend, mentionsEmployee, mentionsTeam, mentionsDepartment, mentionsTask, mentionsProject
- "source" (VARCHAR): Source entity id
- "target" (VARCHAR): Target entity id
- "temporality" (VARCHAR): Temporal nature (ongoing, start, end)
- "beginning" (DATE): Start date
- "end" (DATE): End date

TEMPORAL SEMANTICS - CRITICAL:
- beginning = NULL → Extends INFINITELY INTO THE PAST
- end = NULL → Extends INFINITELY INTO THE FUTURE
- Both NULL → ALWAYS TRUE (no temporal bounds)"""

_DOCUMENT_SCHEMA = """gold_document:
- "id" (VARCHAR): Unique identifier (filename)
- "date" (DATE): Document date
- "sender_alias" (TEXT): Sender's alias
- "sender_name" (TEXT): Sender's full name
- "sender_email" (TEXT): Sender's email address
- "recipient_aliases" (TEXT[]): Array of recipient aliases
- "recipient_names" (TEXT[]): Array of recipient names
- "recipient_emails" (TEXT[]): Array of recipient emails
- "represents_meeting" (VARCHAR): Meeting ID if represents a meeting"""
_DOCUMENT_SCHEMA_BODY = _DOCUMENT_SCHEMA.removeprefix("gold_document:\n")
_TOPIC_DOCUMENT_SCHEMA_BODY = """- "id" (TEXT): Unique document identifier (filename)
- "date" (DATE): Document date
- "sender_alias" (TEXT): Sender's alias
- "sender_name" (TEXT): Sender's full name
- "sender_email" (TEXT): Sender's email address
- "recipient_aliases" (TEXT[]): Array of recipient aliases
- "recipient_names" (TEXT[]): Array of recipient names
- "recipient_emails" (TEXT[]): Array of recipient emails
- "represents_meeting" (VARCHAR): Meeting ID if document represents a meeting"""

_KB_EXAMPLES = """Common query patterns:

Simple entity queries:
- SELECT name FROM gold_employee WHERE name LIKE '%Smith%'
- SELECT name, aliases FROM gold_company

Relationship queries:
- Find who works at a company:
  SELECT e.name FROM gold_employee e
  JOIN gold_relationship r ON e.id = r.source
  JOIN gold_company c ON r.target = c.id
  WHERE r.type = 'worksAt' AND c.name = 'BioCure Solutions GmbH'

- Find team members:
  SELECT e.name FROM gold_employee e
  JOIN gold_relationship r ON e.id = r.source
  JOIN gold_team t ON r.target = t.id
  WHERE r.type = 'memberOf' AND t.name = 'Engineering Team'

Temporal relationship queries:
- Find who worked at a company on a specific date:
  SELECT e.name FROM gold_employee e
  JOIN gold_relationship r ON e.id = r.source
  JOIN gold_company c ON r.target = c.id
  WHERE r.type = 'worksAt' AND c.name = 'BioCure Solutions GmbH'
  AND (r.beginning IS NULL OR r.beginning <= '2024-03-15')
  AND (r.end IS NULL OR r.end >= '2024-03-15')"""

_DOCUMENT_EXAMPLES = """Common query patterns:
- Find documents by sender: SELECT * FROM gold_document WHERE "sender_name" LIKE '%John%'
- Find documents by date: SELECT * FROM gold_document WHERE "date" BETWEEN '2023-01-01' AND '2023-12-31'
- Find documents with recipient: SELECT * FROM gold_document WHERE 'john@example.com' = ANY("recipient_emails")
- Find meeting documents: SELECT * FROM gold_document WHERE "represents_meeting" IS NOT NULL"""

_INTEGRATED_EXAMPLES = """Common query patterns:

Entity queries:
- Find employees: SELECT "name" FROM gold_employee WHERE "name" LIKE '%Smith%'
- Find relationships: SELECT * FROM gold_relationship WHERE "type" = 'worksAt'

Document queries:
- Find documents by sender: SELECT * FROM gold_document WHERE "sender_name" LIKE '%John%'
- Find documents by date: SELECT * FROM gold_document WHERE "date" BETWEEN '2023-01-01' AND '2023-12-31'
- Find meeting documents: SELECT * FROM gold_document WHERE "represents_meeting" IS NOT NULL
- Find documents with recipient: SELECT * FROM gold_document WHERE 'alice@example.com' = ANY("recipient_emails")

Temporal relationship queries:
- Find who worked at company on specific date:
  SELECT e.name FROM gold_employee e
  JOIN gold_relationship r ON e.id = r.source
  JOIN gold_company c ON r.target = c.id
  WHERE r.type = 'worksAt' AND c.name = 'BioCure'
  AND (r.beginning IS NULL OR r.beginning <= '2024-03-15')
  AND (r.end IS NULL OR r.end >= '2024-03-15')

Integrated queries (combining entities and documents):
- Find documents from employee:
  SELECT d.* FROM gold_document d
  WHERE d.sender_email IN (SELECT work_email FROM gold_employee WHERE name = 'John Smith')

- Find documents from team members on date:
  SELECT d.* FROM gold_document d
  WHERE d.sender_email IN (
    SELECT e.work_email FROM gold_employee e
    JOIN gold_relationship r ON e.id = r.source
    WHERE r.type = 'memberOf' AND r.target = 'team_id'
    AND (r.beginning IS NULL OR r.beginning <= d.date)
    AND (r.end IS NULL OR r.end >= d.date)
  )"""


def system_prompt(method: str, task: str, *, tool_limit: int) -> str:
    if method == "rag":
        return RAG_SYSTEM.format(tool_limit=tool_limit)
    constraint = (
        "BENCHMARK CONSTRAINTS:\n"
        f"- You are limited to {tool_limit} tool calls maximum. Use them wisely."
    )
    if task == "kb_qa":
        return f"""You are an assistant specializing in company knowledge bases containing entities and relationships.

Your job is to:
1. Write a SQL query that suits the user's request
2. Execute the query using the execute_sql tool to show the actual results

IMPORTANT RULES:
1. Table names: gold_employee, gold_company, gold_task, gold_project, gold_department, gold_team, gold_meeting, gold_relationship
2. Column names must be enclosed in double quotes (e.g., "hasName", "workEmail")
3. Column names are case-sensitive - use exact capitalization as shown in schema
4. Use proper PostgreSQL syntax with LIKE for text searching
5. For relationship, columns like "works_on", "member_of" contain lists/arrays of IDs
6. Always use the execute_sql tool to run your query and show results to the user

{constraint}

Database schema:

{_ENTITY_SCHEMA}

{_KB_EXAMPLES}"""
    if task == "topic_qa":
        return f"""You are a document analysis assistant specializing in company document repositories.

Your job is to:
1. Write a SQL query that suits the user's request about documents
2. Execute the query using the execute_sql tool to show the actual results

IMPORTANT RULES:
1. Table name: gold_document
2. Column names must be enclosed in double quotes (e.g., "id", "filename", "topic")
3. Column names are case-sensitive - use exact capitalization as shown in schema
4. Use proper PostgreSQL syntax with LIKE for text searching
5. Always use the execute_sql tool to run your query and show results to the user

{constraint}
- Return results in the exact format specified by the user.

Database schema for documents table:
{_TOPIC_DOCUMENT_SCHEMA_BODY}

{_DOCUMENT_EXAMPLES}"""
    if task != "integrated_qa":
        raise ValueError("Unknown QA task")
    return f"""You are an assistant specializing in integrated company knowledge bases containing entities, relationships, and documents.

Your job is to:
1. Analyze the user's question to determine whether it requires querying structured data (entities/relationships) or documents
2. Write appropriate SQL queries to access the relevant information
3. Execute queries using the execute_sql tool to show actual results
4. Synthesize information from both structured data and documents when needed

IMPORTANT RULES:
1. Table names: gold_employee, gold_company, gold_task, gold_project, gold_department, gold_team, gold_meeting, gold_relationship, gold_document
2. Column names must be enclosed in double quotes (e.g., "hasName", "workEmail", "topic")
3. Column names are case-sensitive - use exact capitalization as shown in schema
4. Use proper PostgreSQL syntax with LIKE for text searching
5. For relationships table, columns like "works_on", "member_of" contain lists/arrays of IDs
6. The documents table contains company communications and can be queried separately or joined with entity data
7. Always use the execute_sql tool to run your query and show results to the user

{constraint}
- Return results in the exact format specified by the user.

Database schema:

{_ENTITY_SCHEMA}

DOCUMENTS:

{_DOCUMENT_SCHEMA}

{_INTEGRATED_EXAMPLES}"""


def question_prompt(method: str, task: str, question: dict[str, Any]) -> str:
    description = _DESCRIPTIONS["rag" if method == "rag" else task]
    return f"""You are answering a question about {description}.

When answering the question, it is crucial to follow these instructions:
{_ANSWER_INSTRUCTIONS[question["answer_type"]]}
REPLY WITH NOTHING ELSE EXCEPT FOR YOUR ANSWER!

Here are examples of questions and answers for this type of evaluation:

{_EXAMPLES[task]}

Answer the following question:
{question["question"]}"""
