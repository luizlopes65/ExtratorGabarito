#!/usr/bin/env python3
"""
Compile and manage batch extraction results with banco.csv metadata.

This script:
1. Processes batch extraction results from multiple answer sheets
2. Links student answers with question metadata from banco.csv
3. Creates a master table with all data
4. Prepares data for heatmap generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


class ResultsCompiler:
    """Manages compilation of extraction results with metadata."""
    
    def __init__(self, banco_csv_path: str = "banco.csv"):
        """
        Initialize the compiler.
        
        Args:
            banco_csv_path: Path to the question metadata CSV
        """
        self.banco_df = pd.read_csv(banco_csv_path)
        self.master_data = []
        
    def load_extraction_results(self, results_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load all extraction results from a batch processing directory.
        
        Args:
            results_dir: Directory containing batch results
            
        Returns:
            Dictionary mapping image names to their result DataFrames
        """
        results_path = Path(results_dir)
        extractions = {}
        
        # Each image has its own subdirectory with resultado.csv
        for image_dir in results_path.iterdir():
            if image_dir.is_dir():
                result_file = image_dir / "resultado.csv"
                if result_file.exists():
                    df = pd.read_csv(result_file)
                    extractions[image_dir.name] = df
                    print(f"Loaded: {image_dir.name} ({len(df)} students)")
        
        return extractions
    
    def extract_question_number(self, question_id: str) -> str:
        """
        Extract question number from various formats.
        
        Examples:
            "01.pdf" -> "1"
            "39 - V1.pdf" -> "39"
            "64-A" -> "64"
        """
        # Remove .pdf extension
        q = question_id.replace('.pdf', '')
        
        # Remove version indicators (V1, V2)
        q = q.split(' - ')[0].strip()
        
        # Extract just the number part
        import re
        match = re.match(r'(\d+)', q)
        if match:
            return match.group(1)
        return q
    
    def get_question_metadata(self, question_number: str) -> Dict:
        """
        Get metadata for a specific question from banco.csv.
        
        Args:
            question_number: Question number (e.g., "1", "35", "64")
            
        Returns:
            Dictionary with question metadata
        """
        # Try to find the question in banco.csv
        # The Questão column has format like "01.pdf", "35.pdf", etc.
        question_id = f"{int(question_number):02d}.pdf"
        
        matches = self.banco_df[self.banco_df['Questão'] == question_id]
        
        if len(matches) == 0:
            # Try without leading zero
            question_id = f"{int(question_number)}.pdf"
            matches = self.banco_df[self.banco_df['Questão'] == question_id]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            return {
                'question_id': question_id,
                'grade': row['Ano Escolar'],
                'subject': row['Assunto'],
                'items': row['Itens'] if pd.notna(row['Itens']) else '',
                'metadata': row.to_dict()
            }
        
        return {
            'question_id': question_number,
            'grade': 'Unknown',
            'subject': 'Unknown',
            'items': '',
            'metadata': {}
        }
    
    def compile_single_sheet(self, image_name: str, df: pd.DataFrame) -> List[Dict]:
        """
        Compile data from a single answer sheet.
        
        Args:
            image_name: Name of the image/sheet
            df: DataFrame with extraction results
            
        Returns:
            List of records with student answers and metadata
        """
        records = []
        
        # Get question columns (all except 'Nome')
        question_cols = [col for col in df.columns if col != 'Nome']
        
        for _, row in df.iterrows():
            student_name = row['Nome']
            
            for question_col in question_cols:
                answer = row[question_col]
                
                # Get question metadata
                metadata = self.get_question_metadata(question_col)
                
                record = {
                    'sheet_id': image_name,
                    'student_name': student_name,
                    'question_number': question_col,
                    'answer': answer if pd.notna(answer) else '',
                    'grade': metadata['grade'],
                    'subject': metadata['subject'],
                    'items': metadata['items'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add all metadata columns
                for key, value in metadata['metadata'].items():
                    if key not in record and pd.notna(value):
                        record[f'meta_{key}'] = value
                
                records.append(record)
        
        return records
    
    def compile_all_results(self, results_dir: str) -> pd.DataFrame:
        """
        Compile all extraction results into a master table.
        
        Args:
            results_dir: Directory containing batch results
            
        Returns:
            Master DataFrame with all data
        """
        print("\n" + "="*80)
        print("COMPILING BATCH RESULTS")
        print("="*80)
        
        # Load all extractions
        extractions = self.load_extraction_results(results_dir)
        
        if not extractions:
            print("No extraction results found!")
            return pd.DataFrame()
        
        print(f"\nFound {len(extractions)} answer sheets")
        
        # Compile each sheet
        all_records = []
        for image_name, df in extractions.items():
            print(f"Processing: {image_name}")
            records = self.compile_single_sheet(image_name, df)
            all_records.extend(records)
            print(f"  Added {len(records)} records")
        
        # Create master DataFrame
        master_df = pd.DataFrame(all_records)
        
        print(f"\n✓ Master table created: {len(master_df)} total records")
        print(f"  Unique students: {master_df['student_name'].nunique()}")
        print(f"  Unique questions: {master_df['question_number'].nunique()}")
        print(f"  Unique sheets: {master_df['sheet_id'].nunique()}")
        
        return master_df
    
    def create_student_summary(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary table with one row per student.
        
        Args:
            master_df: Master DataFrame with all records
            
        Returns:
            Summary DataFrame with student performance
        """
        summary_records = []
        
        for student in master_df['student_name'].unique():
            student_data = master_df[master_df['student_name'] == student]
            
            # Count answers by subject
            subject_counts = student_data.groupby('subject')['answer'].apply(
                lambda x: (x != '').sum()
            ).to_dict()
            
            record = {
                'student_name': student,
                'total_questions': len(student_data),
                'answered': (student_data['answer'] != '').sum(),
                'blank': (student_data['answer'] == '').sum(),
                'sheets_completed': student_data['sheet_id'].nunique()
            }
            
            # Add subject-specific counts
            for subject, count in subject_counts.items():
                record[f'answered_{subject}'] = count
            
            summary_records.append(record)
        
        return pd.DataFrame(summary_records)
    
    def create_question_summary(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary table with one row per question.
        
        Args:
            master_df: Master DataFrame with all records
            
        Returns:
            Summary DataFrame with question statistics
        """
        summary_records = []
        
        for question in master_df['question_number'].unique():
            question_data = master_df[master_df['question_number'] == question]
            
            # Get metadata (should be same for all rows of this question)
            metadata = question_data.iloc[0]
            
            # Count answer distribution
            answer_counts = question_data['answer'].value_counts().to_dict()
            
            record = {
                'question_number': question,
                'grade': metadata['grade'],
                'subject': metadata['subject'],
                'items': metadata['items'],
                'total_responses': len(question_data),
                'answered': (question_data['answer'] != '').sum(),
                'blank': (question_data['answer'] == '').sum(),
            }
            
            # Add answer distribution
            for answer, count in answer_counts.items():
                if answer:  # Skip blank answers
                    record[f'answer_{answer}'] = count
            
            summary_records.append(record)
        
        return pd.DataFrame(summary_records)
    
    def save_compiled_results(self, master_df: pd.DataFrame, output_dir: str = "resultados/compiled"):
        """
        Save all compiled results to files.
        
        Args:
            master_df: Master DataFrame with all data
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save master table
        master_file = output_path / f"master_table_{timestamp}.csv"
        master_df.to_csv(master_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ Master table saved: {master_file}")
        
        # Save student summary
        student_summary = self.create_student_summary(master_df)
        student_file = output_path / f"student_summary_{timestamp}.csv"
        student_summary.to_csv(student_file, index=False, encoding='utf-8-sig')
        print(f"✓ Student summary saved: {student_file}")
        
        # Save question summary
        question_summary = self.create_question_summary(master_df)
        question_file = output_path / f"question_summary_{timestamp}.csv"
        question_summary.to_csv(question_file, index=False, encoding='utf-8-sig')
        print(f"✓ Question summary saved: {question_file}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_records': len(master_df),
            'unique_students': int(master_df['student_name'].nunique()),
            'unique_questions': int(master_df['question_number'].nunique()),
            'unique_sheets': int(master_df['sheet_id'].nunique()),
            'files': {
                'master': str(master_file),
                'students': str(student_file),
                'questions': str(question_file)
            }
        }
        
        metadata_file = output_path / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Metadata saved: {metadata_file}")
        
        return {
            'master': master_file,
            'students': student_file,
            'questions': question_file,
            'metadata': metadata_file
        }


def main():
    """Main entry point for compilation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile batch extraction results with metadata"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='resultados/batch',
        help='Directory containing batch extraction results'
    )
    parser.add_argument(
        '--banco',
        type=str,
        default='banco.csv',
        help='Path to banco.csv with question metadata'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='resultados/compiled',
        help='Directory to save compiled results'
    )
    
    args = parser.parse_args()
    
    # Create compiler
    compiler = ResultsCompiler(banco_csv_path=args.banco)
    
    # Compile all results
    master_df = compiler.compile_all_results(args.results_dir)
    
    if len(master_df) > 0:
        # Save results
        files = compiler.save_compiled_results(master_df, args.output_dir)
        
        print("\n" + "="*80)
        print("COMPILATION COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        for key, path in files.items():
            print(f"  {key}: {path}")
    else:
        print("\n❌ No data to compile")


if __name__ == "__main__":
    main()

# Made with Bob
