"""Minimal private non-executing credential loader. No logging or network."""
from pathlib import Path

ALIASES={'openai':'chatgpt.txt','deepseek':'deepseek.txt','ncbi':'NCBI.txt',
         'openalex':'OPENALEX.txt','github':'GITHUB_fine_grained 2.txt'}


def load_secret(root,provider):
    if provider not in ALIASES:raise ValueError('Unidentified provider is disabled')
    path=Path(root)/'apis'/ALIASES[provider]
    if not path.is_file():return None
    lines=[line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(lines)!=1:raise ValueError('Credential must be one nonempty data line')
    value=lines[0]
    assignment_names={'KEY','TOKEN','API_KEY','OPENAI_API_KEY','DEEPSEEK_API_KEY','NCBI_API_KEY','OPENALEX_API_KEY','GITHUB_TOKEN'}
    if '=' in value and value.split('=',1)[0] in assignment_names:value=value.split('=',1)[1].strip()
    if not value or any(c.isspace() for c in value):raise ValueError('Invalid credential format')
    return value
