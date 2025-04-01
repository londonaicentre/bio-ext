# Elastic Database Directory

This is a shared resource for what clinically significant documents exist in which document index, and the corresponding query required to discover them.The type of document is incredibly important, as it defines the context in which a note was written. Pulling data from appropriate documents is also likely to increase the precision of extracted concepts, by reducing unnecessary noise.

## Epic Clinical Notes - Fields of Interest

> * id: Unique ID for the source Elastic index (e.g. "ecn_9283646")
> * document_EpicId: ID of document in Epic
> * activity_Date: Date at which activity occurred
> * document_CreatedWhen: Date at which document was first created
> * document_UpdatedWhen: Date at which document was last updated (note that any edits are discarded)
> * patient_NHSNumber: Patient's NHS number if present
> * patient_DurableKey: Patient's Epic Durable Key
> * activity_EncounterEpicCsn: Link to encounter/activity
> * activity_PatientAdministrativeCategory: Type of patient (e.g. if NHS or Private etc),
> * activity_PatientClass: Type of pathway the patient is on (e.g. Outpatient or Inpatient etc),
> * activity_Type: Type of activity, (e.g. Clinic/practice Visit)
> * activity_VisitClass: How visit occurred (e.g. Outpatient F2F etc)
> * activity_VisitType: Granular description of visit (e.g. 2WW, WALK IN URGENT etc)
> * activity_DepartmentSpecialty: Which department the activity happened in
> * activity_ChiefComplaint: Main problem identified for encounter/activity
> * document_Content: Main Body of text
> * document_Name: Name of document class (e.g. Progress Notes)
> * document_AuthorType: Role of author (e.g. Specialty Doctor)
> * document_Service: Specialism linked to document
> * document_Status: Whether document is signed off or not

## Epic Clinical Notes - Outpatient Notes

These are records taken during outpatient visit consultations. They usually contain a summary of the patient's primary diagnoses and background, as well as a review of active issues and a plan. Further filtering on activity_DepartmentSpecialty and document_Service can help discover outpatient encounters for specific patient pathways (e.g. Cancer)

```json
{
    "_source": [
        "id",
        "document_EpicId",
        "activity_Date",
        "document_CreatedWhen",
        "document_UpdatedWhen",
        "patient_NHSNumber",
        "patient_DurableKey",
        "activity_EncounterEpicCsn",
        "activity_PatientAdministrativeCategory",
        "activity_PatientClass",
        "activity_Type",
        "activity_VisitClass",
        "activity_VisitType",
        "activity_DepartmentSpecialty",
        "activity_ChiefComplaint",
        "document_Content",
        "document_Name",
        "document_AuthorType",
        "document_Service",
        "document_Status"
    ],
    "query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "match_phrase": {
                        "activity_Type": "Clinic/Practice Visit"
                    }
                }
            ],
            "should": [],
            "must_not": [
                {
                    "match_phrase": {
                        "document_Name.keyword": "Appointment Note"
                    }
                },
                {
                    "match_phrase": {
                        "document_Name.keyword": "Nursing Note"
                    }
                }
            ]
        }
    }
}
```

## brca documents
```json
{
    "_source": [
        "id",
        "document_Comment",
	"document_Content",
	"document_ContentType",
	"document_CreatedWhen",
	"document_CareProviderType",
	"document_FileName",
	"document_Name",
	"document_Type",
	"document_Subject",
	"patient_DateOfBirth",
	"patient_FirstName",
	"patient_LastName",
	"patient_Gender",
	"patient_GlobalConsultant",
	"patient_HospitalNumber",
	"patient_HospitalNumber_GSTT",
	"patient_MRN",
	"patient_NhsNumber"
	
    ],
    "query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "match_phrase": {
                        "document_Content": "breast cancer"
                    }
                }
            ],
            "should": [],
            "must_not": [
                {
                    "match_phrase": {
                        "document_Name.keyword": "Appointment Note"
                    }
                },
                {
                    "match_phrase": {
                        "document_Name.keyword": "Nursing Note"
                    }
                }
            ]
        }
    }
}
```

## cancer clinic letters

```json
{
  "source": [
    "document_Type",
    "document_Name",
    "document_EncouterDate",
    "document_Content",
    "document_CreatedWhen",
    "document_UpdatedWhen",
    "patient_HospitalNumber_GSTT",
    "patient_HospitalNumber_KCH",
    "patient_RadiotherapyConsultant",
    "patient_GlobalConsultant",
    "patient_SourceId",
    "patient_NhsNumber",
    "patient_FirstName",
    "patient_MiddleName",
    "patient_LastName",
    "patient_Gender",
    "patient_DateOfBirth",
    "patient_Ethnicity"
  ],
  "query": {
    "bool": {
      "must": [],
      "filter":[
        {
          "terms": {
            "document_Name.keyword": [
              "Clinical-Out Pt",
              "Consultation letter",
              "Clinical-MDM",
              "Clinical-In Pt",
              "Clinical-Key Data"
            ]
          }
        }
      ]
    }
  }
}
```
