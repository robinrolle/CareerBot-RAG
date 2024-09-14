'use client';

import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import { Tabs, Tab } from "@nextui-org/react";
import SuggestionsSection from './components/SuggestionsSection';
import SelectionsSection from './components/SelectionsSection';
import './style/Tabs.css';

export default function Home() {
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  const [skillsOptions, setSkillsOptions] = useState([]);
  const [skillsData, setSkillsData] = useState([]);
  const [initialSkillsData, setInitialSkillsData] = useState([]); 
  const [suggestedSkills, setSuggestedSkills] = useState([]);
  const [initialSuggestedSkills, setInitialSuggestedSkills] = useState([]); 

  const [occupationsOptions, setOccupationsOptions] = useState([]);
  const [occupationsData, setOccupationsData] = useState([]);
  const [initialOccupationsData, setInitialOccupationsData] = useState([]); 
  const [suggestedOccupations, setSuggestedOccupations] = useState([]);
  const [initialSuggestedOccupations, setInitialSuggestedOccupations] = useState([]);

  const [activeTab, setActiveTab] = useState('skills');

  // load options
  useEffect(() => {
    const loadSkillsOptions = async () => {
      try {
        const response = await fetch('/data/skills_options.json');  
        const data = await response.json();
        setSkillsOptions(data);
        console.log('Skills options:', data); 
      } catch (error) {
        console.error("Error at loading skills options :", error);
      }
    };

    const loadOccupationsOptions = async () => {
      try {
        const response = await fetch('/data/occupations_options.json');
        const data = await response.json();
        setOccupationsOptions(data);
        console.log('Occupations options:', data); 
      } catch (error) {
        console.error("Error at loading occupations options :", error);
      }
    };

    loadSkillsOptions();
    loadOccupationsOptions();
  }, []);

  const handleFileUpload = (filename) => {
    setUploadedFilename(filename);
    setAnalyzed(false);
  };

  const handleFileRemove = () => {
    setUploadedFilename(null);
    setAnalyzed(false);
    setSkillsData([]);
    setSuggestedSkills([]);
    setOccupationsData([]);
    setSuggestedOccupations([]);
  };

  const handleSubmit = async () => {
    if (!uploadedFilename) return;

    setLoading(true);

    try {
      const response = await fetch(`http://localhost:8000/process?filename=${uploadedFilename}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error('Failed to process the file');
      }

      const data = await response.json();

      // Update current and initial states
      setSkillsData(data.graded_skills);
      setInitialSkillsData(data.graded_skills);
      setSuggestedSkills(data.suggestions.suggested_skills_ids);
      setInitialSuggestedSkills(data.suggestions.suggested_skills_ids);

      setOccupationsData(data.graded_occupations);
      setInitialOccupationsData(data.graded_occupations);
      setSuggestedOccupations(data.suggestions.suggested_occupations_ids);
      setInitialSuggestedOccupations(data.suggestions.suggested_occupations_ids);

      setAnalyzed(true);
    } catch (error) {
      console.error('Error during processing:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSkillsChange = (newSkills) => {
    setSkillsData(prevSkills => {
      const existingSkills = prevSkills.filter(skill => newSkills.includes(skill.id));
      const addedSkills = newSkills
        .filter(id => !prevSkills.some(skill => skill.id === id))
        .map(id => ({ id, item: skillsOptions.find(opt => opt.value === id)?.label || id, relevance: null }));
      return [...existingSkills, ...addedSkills];
    });
    // Mise à jour des suggestions
    setSuggestedSkills(prevSuggestions => prevSuggestions.filter(id => !newSkills.includes(id)));
  };

  const handleOccupationsChange = (newOccupations) => {
    setOccupationsData(prevOccupations => {
      const existingOccupations = prevOccupations.filter(occupation => newOccupations.includes(occupation.id));
      const addedOccupations = newOccupations
        .filter(id => !prevOccupations.some(occupation => occupation.id === id))
        .map(id => ({ id, item: occupationsOptions.find(opt => opt.value === id)?.label || id, relevance: null }));
      return [...existingOccupations, ...addedOccupations];
    });
    // Mise à jour des suggestions
    setSuggestedOccupations(prevSuggestions => prevSuggestions.filter(id => !newOccupations.includes(id)));
  };

  const removeSkill = (skillId) => {
    setSkillsData(prevSkills => prevSkills.filter(skill => skill.id !== skillId));
  };

  const removeOccupation = (occupationId) => {
    setOccupationsData(prevOccupations => prevOccupations.filter(occupation => occupation.id !== occupationId));
  };

  const resetSelections = (type) => {
    if (type === 'skills') {
      setSkillsData([...initialSkillsData]);
      setSuggestedSkills([...initialSuggestedSkills]);
    } else if (type === 'occupations') {
      setOccupationsData([...initialOccupationsData]);
      setSuggestedOccupations([...initialSuggestedOccupations]);
    }
  };

  const handleTabChange = (tabKey) => {
    setActiveTab(tabKey);
  };

  const handleSuggestSkills = async () => {
    setLoading(true);
  
    try {
      const response = await fetch('http://localhost:8000/suggestions/skills', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(skillsData.map(skill => skill.id)),
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch skill suggestions');
      }
  
      const data = await response.json();
      setSuggestedSkills(data.suggested_skills_ids || []);
  
    } catch (error) {
      console.error('Error fetching skill suggestions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestOccupations = async () => {
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/suggestions/occupations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(occupationsData.map(occupation => occupation.id)),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch occupation suggestions');
      }

      const data = await response.json();
      setSuggestedOccupations(data.suggested_occupations_ids || []);

    } catch (error) {
      console.error('Error fetching occupation suggestions:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Head>
        {/* TODO add title and logo */}
      </Head>

      <header className="flex justify-between items-center p-4 bg-white">
        <div>
          <Image src="/media/3amk_full.png" alt="Logo" width={100} height={75} />
        </div>
        <nav className="flex space-x-4">
          <Link href="/about" className="text-gray-700 hover:text-gray-900">
            About
          </Link>
          <Link href="/contact" className="text-gray-700 hover:text-gray-900">
            Contact
          </Link>
        </nav>
      </header>

      <main className="container mx-auto px-16 py-16">
        <h1 className="text-5xl font-bold text-center mb-4">CareerBot</h1>
        <p className="text-xl text-center mb-10">Analyze your resume and build your profile!</p>

        <div className="flex flex-col items-center">
          <div className="w-full max-w-md mb-4">
            <FileUpload onFileUpload={handleFileUpload} onFileRemove={handleFileRemove} />
          </div>

          {uploadedFilename && (
            <button
              className={`w-full p-4 text-white rounded-xl shadow max-w-md mb-8
                        ${loading || analyzed ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'}`}
              onClick={handleSubmit}
              disabled={loading || analyzed}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          )}
        </div>

        {loading && !analyzed && (
          <div className="flex justify-center items-center h-64">
            <div className="border-gray-300 h-20 w-20 animate-spin rounded-full border-8 border-t-blue-600"></div>
          </div>
        )}

        {analyzed && (
          <div className="mx-auto flex flex-col items-start">
            <h2 className="text-4xl font-semibold">Your profile</h2>
            <Tabs key="types" className="customTabs" selectedKey={activeTab} onSelectionChange={handleTabChange} disableAnimation={true}>
              <Tab key="skills" title="Skills" className="customTab" />
              <Tab key="occupations" title="Occupations" className="customTab" />
            </Tabs>

            {activeTab === 'skills' && (
              <div className="skills-container w-full">
                <SelectionsSection
                  title="Identified"
                  selections={skillsData}
                  options={skillsOptions}
                  placeholder="Select skills..."
                  onChange={handleSkillsChange}
                  onRemove={removeSkill}
                  onReset={() => resetSelections('skills')}
                />
                <hr className="mb-4 mt-4"/>
                <SuggestionsSection
                  title="Suggestions"
                  suggestions={suggestedSkills}
                  options={skillsOptions}
                  onSelect={(selectedOption) => {
                    handleSkillsChange([...skillsData.map(skill => skill.id), selectedOption.value]);
                  }}
                  onSuggest={handleSuggestSkills}
                />
              </div>
            )}

            {activeTab === 'occupations' && (
              <div className="occupations-container w-full">
                <SelectionsSection
                  title="Identified"
                  selections={occupationsData}
                  options={occupationsOptions}
                  placeholder="Select occupations..."
                  onChange={handleOccupationsChange}
                  onRemove={removeOccupation}
                  onReset={() => resetSelections('occupations')}
                />
                <hr className="mb-4 mt-4"/>
                <SuggestionsSection
                  title="Suggestions"
                  suggestions={suggestedOccupations}
                  options={occupationsOptions}
                  onSelect={(selectedOption) => {
                    handleOccupationsChange([...occupationsData.map(occupation => occupation.id), selectedOption.value]);
                  }}
                  onSuggest={handleSuggestOccupations}
                />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}