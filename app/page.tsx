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
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [suggestedSkills, setSuggestedSkills] = useState([]);
  const [responseSkills, setResponseSkills] = useState([]);

  const [occupationsOptions, setOccupationsOptions] = useState([]);
  const [selectedOccupations, setSelectedOccupations] = useState([]);
  const [suggestedOccupations, setSuggestedOccupations] = useState([]);
  const [responseOccupations, setResponseOccupations] = useState([]);

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

        setResponseSkills(data.skills_ids || []);
        setSuggestedSkills(data.skills_ids || []);
        setResponseOccupations(data.occupations_ids || []);
        setSuggestedOccupations(data.occupations_ids || []);

        setAnalyzed(true);
    } catch (error) {
        console.error('Error during processing:', error);
    } finally {
        setLoading(false);
    }
};

  const handleTabChange = (tabKey) => {
    setActiveTab(tabKey);
  };

  // Function to handle adding a skill from suggestions to selections
  const addSkillFromSuggestions = (skillValue) => {
    const skill = skillsOptions.find(opt => opt.value === skillValue);
    if (skill) {
      setSelectedSkills(prevSelected => [...prevSelected, skill]);
      setSuggestedSkills(prevSuggested => prevSuggested.filter(s => s !== skillValue));
    }
  };

  // Function to handle removing a skill from selections or suggestions
  const removeSkill = (option, isSelection) => {
    if (isSelection) {
      setSelectedSkills(prevSelected => prevSelected.filter(skill => skill.value !== option.value));
    } else {
      setSuggestedSkills(prevSuggested => prevSuggested.filter(skill => skill !== option.value));
    }
  };

  const resetSuggestedSkills = () => {
    const resetSkills = responseSkills.filter(skill =>
      !selectedSkills.some(selected => selected.value === skill)
    );
    setSuggestedSkills(resetSkills);
  };

  const clearSelectedSkills = () => {
    setSelectedSkills([]);
  };

  // Similarly for occupations
  const addOccupationFromSuggestions = (occupationValue) => {
    const occupation = occupationsOptions.find(opt => opt.value === occupationValue);
    if (occupation) {
      setSelectedOccupations(prevSelected => [...prevSelected, occupation]);
      setSuggestedOccupations(prevSuggested => prevSuggested.filter(o => o !== occupationValue));
    }
  };

  const removeOccupation = (option, isSelection) => {
    if (isSelection) {
      setSelectedOccupations(prevSelected => prevSelected.filter(occupation => occupation.value !== option.value));
    } else {
      setSuggestedOccupations(prevSuggested => prevSuggested.filter(occupation => occupation !== option.value));
    }
  };

  const resetSuggestedOccupations = () => {
    const resetOccupations = responseOccupations.filter(occupation =>
      !selectedOccupations.some(selected => selected.value === occupation)
    );
    setSuggestedOccupations(resetOccupations);
  };

  const clearSelectedOccupations = () => {
    setSelectedOccupations([]);
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
              className={`w-full p-4 text-white rounded-xl shadow max-w-md 
                        ${loading || analyzed ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'}`}
              onClick={handleSubmit}
              disabled={loading || analyzed}
            >
              {loading ? (
                <>
                  Analyzing...
                </>
              ) : (
                'Analyze'
              )}
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
                <SuggestionsSection
                  title="Bot suggestions"
                  suggestions={suggestedSkills}
                  options={skillsOptions}
                  onReset={resetSuggestedSkills}
                  onAdd={addSkillFromSuggestions}
                  onRemove={removeSkill}
                />
                <hr className="my-4" />
                <SelectionsSection
                  title="Selections"
                  selections={selectedSkills}
                  options={skillsOptions.filter(opt => !selectedSkills.some(skill => skill.value === opt.value) && !suggestedSkills.includes(opt.value))}
                  placeholder="Select skills..."
                  onChange={setSelectedSkills}
                  onClear={clearSelectedSkills}
                  onRemove={removeSkill}
                />
              </div>
            )}

            {activeTab === 'occupations' && (
              <div className="occupations-container w-full">
                <SuggestionsSection
                  title="Bot suggestions"
                  suggestions={suggestedOccupations}
                  options={occupationsOptions}
                  onReset={resetSuggestedOccupations}
                  onAdd={addOccupationFromSuggestions}
                  onRemove={removeOccupation}
                />
                <hr className="my-4" />
                <SelectionsSection
                  title="Selections"
                  selections={selectedOccupations}
                  options={occupationsOptions.filter(opt => !selectedOccupations.some(occupation => occupation.value === opt.value) && !suggestedOccupations.includes(opt.value))}
                  placeholder="Select occupations..."
                  onChange={setSelectedOccupations}
                  onClear={clearSelectedOccupations}
                  onRemove={removeOccupation}
                />

              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
