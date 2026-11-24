"""Persist current verified results, without resetting accounting or historical evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import write_json,sha256,source_identity,environment

def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf-8-sig'))

def main():
    stats=read('results/statistics.json');tests=read('reports/tests.json')
    audit=read('reports/manuscript_audit.json');visual=read('manuscript/visual_review.json')
    assert tests['status']=='PASS'
    assert all(sha256(ROOT/name)==digest for name,digest in tests['source_hashes'].items())
    assert stats['candidate_discrete_pass']==108 and stats['legacy_reported_converged']==90
    assert audit['status']=='PASS_LOCAL_CHECKS' and audit['actual_DOCX_visual_review']=='PASS_CURRENT_HASHES'
    for key,relative in [('docx_sha256','manuscript/NanoporeOS_manuscript.docx'),
                         ('pdf_sha256','manuscript/NanoporeOS_manuscript.pdf'),('source_sha256','manuscript/manuscript.md')]:
        assert visual[key]==sha256(ROOT/relative)
    build=read('manuscript/build_checks.json')
    build.update(DOCX_render='PASS_ACTUAL_DOCX_WORD16_POPPLER_ALL_PAGES_INSPECTED',
        PDF_render=f"PASS_{visual['pages_inspected']}_PAGES_FROM_ACTUAL_DOCX",visual_review='manuscript/visual_review.json')
    write_json(ROOT/'manuscript/build_checks.json',build)
    checklist=read('manuscript/submission_checklist.json')
    checklist['checks'].update(author_names_affiliations_correspondence='PROVIDED_IN_USER_ORDER',ORCIDs='NOT_PROVIDED_OPTIONAL',
        official_current_requirements='9NOV_RENDERED_AUDIT;21NOV_RECHECK403_AND_BROWSER_DISABLED;FINAL_CHECK_PENDING',
        author_names_affiliations_correspondence_ORCID='AUTHOR_DETAILS_PROVIDED_ORCIDS_NOT_PROVIDED',
        human_contributions_funding_conflicts='UNCONFIRMED',
        citation_and_novelty_coverage=f"{build['references']}_VERIFIED_IDENTITIES_BOUNDED_PRIMARY_EVIDENCE_NO_FIRST_CLAIM",
        manuscript_review='VERIFICATION_EXTENSION_AUDITED; MATCHED_ESTIMATOR_LIMIT_EXPLICIT',DOCX_render=build['DOCX_render'],PDF_render=build['PDF_render'],
        public_code_data_access='HISTORICAL_CREATION_VERIFIED; PUBLIC_EMPTY_AT_LATEST_CHECK; PRIVATE_ACCESS_UNVERIFIED; LICENSE_PENDING',
        charged_wall_continuum_domain='HOLD_REPORTED_AS_NEGATIVE_FINDING_NOT_CONCEALED',
        numerical_claims=str(audit['numerical_claim_rows'])+'_MAPPED_TO_HASHED_EXECUTED_SOURCES')
    if (ROOT/'results/verification_analysis/summary.json').exists():
        checklist['checks']['manufactured_code_verification']='PASS_TWO_SMOOTH_DEVELOPMENT_CASES; LEGACY_HOLD_UNCHANGED'
        checklist['checks']['controlled_qualification']='TWO_SEQUENTIAL_EVALUATION_SETS; POSTHOC_MATCHED_CURRENT_NO_FIELD_ADVANTAGE'
    if (ROOT/'results/verification_v5_analysis/summary.json').exists():
        checklist['checks']['controlled_qualification']='MATCHED24_FIXED_COMBINATIONS;FIELD24ACCURATE_VS_CURRENT12ACCURATE12FIELD_INACCURATE;NO_CERTIFICATION'
        checklist['checks']['manufactured_code_verification']='FOUR_SMOOTH_PROFILES_IN_SEQUENTIAL_STUDIES;INDEPENDENT_FORCING_AND135_NEW_OUTPUTS_AUDITED;PHYSICAL_HOLD_UNCHANGED'
    if (ROOT/'reports/staging_test.json').exists() and read('reports/staging_test.json').get('clean_dependency_install'):
        checklist['checks']['clean_independent_install']='PASS_FRESH_DEPENDENCIES_AND_EDITABLE_EXTRACTED_WINDOWS_CHECKOUT;OTHER_PLATFORMS_UNTESTED'
    write_json(ROOT/'manuscript/submission_checklist.json',checklist)
    ledger=read('orchestration/spending_state.json')
    ledger.update(reason='Historical usage remains unknown; pricing evidence verified but production paid gateway/admission reconciliation is not enabled',
        metadata_checks_report='reports/api_checks.json',paid_calls_enabled=False)
    write_json(ROOT/'orchestration/spending_state.json',ledger)
    config=read('config/project_config.json')
    config.update(active_priority='FINISH_MANUSCRIPT_LOCALLY',server_work='DEFERRED_UNTIL_USER_CONFIRMS_CAPACITY',
        paid_API_enabled=False,current_scientific_gate='NUMERICAL_HOLD_CHARGED_WALL_CONTINUUM_DOMAIN')
    write_json(ROOT/'config/project_config.json',config)
    state=read('orchestration/project_state.json')
    state.update(checkpoint_utc=datetime.now(timezone.utc).isoformat(),status='PARTIAL',test_count=tests['tests_run'],tests='PASS',
        manuscript='EXPANDED_LOCAL_AUTHOR_REVIEW_DRAFT_NOT_SUBMISSION_READY',manuscript_main_words=build['main_text_words'],
        manuscript_references=build['references'],manuscript_figures=build['figure_count'],manuscript_editable_tables=build['tables'],
        DOCX_render=build['DOCX_render'],PDF_render=build['PDF_render'],
        deployment='DEFERRED_BY_USER_PENDING_CAPACITY_CONFIRMATION; NO_SERVER_ACTION_THIS_PASS',
        github=read('reports/github_repositories.json'),API_checks='METADATA_CHECKED_OPENAI_401; NO_PAID_INFERENCE',
        dataset='36_NEUTRAL_FULL_FIELD_ANALYTICAL_RECORDS_QUALIFIED; FAILED_FINEST_TARGET_GUARD_TESTED',
        spending=ledger,environment=environment(),source_sha256=source_identity(),
        manuscript_resume_command=r'& .\scripts\build_manuscript_local.ps1',
        resume_command=r'.\.venv\Scripts\python.exe scripts\resume.py --minutes 120')
    for field,relative in [('private_reproduction_package','reproducibility/package_manifest.json'),
                           ('local_submission_review_package','reproducibility/submission_package_manifest.json'),
                           ('staged_reproduction','reports/staging_test.json')]:
        if (ROOT/relative).exists():state[field]=read(relative)
    if (ROOT/'results/verification_analysis/summary.json').exists():state['controlled_verification']=read('results/verification_analysis/summary.json')
    if (ROOT/'results/verification_v5_analysis/summary.json').exists():state['matched_family_verification']=read('results/verification_v5_analysis/summary.json')
    if (ROOT/'results/driven_refinement_v2_analysis/summary.json').exists():state['directional_driven_reference']=read('results/driven_refinement_v2_analysis/summary.json')
    if (ROOT/'results/verification_v5_timing/audit.json').exists():state['live_policy_timing']=read('results/verification_v5_timing/audit.json')
    if (ROOT/'reports/github_access_check_20261120.json').exists():state['latest_github_access_check']=read('reports/github_access_check_20261120.json')
    write_json(ROOT/'orchestration/project_state.json',state)
    targets=['results/statistics.json','results/matrix/campaign.json','results/references/independent_pb.json',
        'results/performance_summary.json','results/dataset/example_dataset.zip','results/manuscript_v2/qualification_audit.json',
        'manuscript/manuscript.md','manuscript/NanoporeOS_manuscript.docx','manuscript/NanoporeOS_manuscript.pdf',
        'reports/manuscript_audit.json','reports/tests.json','reports/api_checks.json','reports/github_repositories.json']
    targets.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'manuscript/figures').glob('figure[1-9].png'))
    targets.extend(name for name in ['results/verification_v5_analysis/summary.json',
        'results/driven_refinement_v2_analysis/summary.json','results/verification_v5_timing/audit.json',
        'config/verification_v5a_protocol.json','config/driven_refinement_ordered_protocol.json',
        'manuscript/literature_db.json','manuscript/numerical_claims.csv','manuscript/claim_evidence.csv'] if (ROOT/name).exists())
    write_json(ROOT/'reports/artifact_hashes.json',dict(files={name:sha256(ROOT/name) for name in targets}))
    print(json.dumps(dict(status='CHECKPOINT_SAVED',tests=tests['tests_run'],pages=visual['pages_inspected'],
        references=build['references'],figures=build['figure_count'],tables=build['tables'],scientific_gate=state['scientific_gate'])))

if __name__=='__main__':main()
