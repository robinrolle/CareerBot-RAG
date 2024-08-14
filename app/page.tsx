'use client';

import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import React, { useState } from 'react';
import FileUpload from './FileUpload';
import { Tabs, Tab, Card, CardHeader, CardBody, Button } from "@nextui-org/react";
import SearchBar from './SearchBar';
import { Chip } from "@nextui-org/react";
import './style/Tabs.css';

export default function Home() {
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  // Skills related state
  const [skillsOptions] = useState([
    { value: 'skill-1', label: 'Skill 1' },
    { value: 'skill-2', label: 'Skill 2' },
    { value: 'skill-3', label: 'Skill 3' },
    { value: 'skill-4', label: 'Skill 4' },
    // Add more options as needed
  ]);
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [suggestedSkills, setSuggestedSkills] = useState([]);
  const [responseSkills, setResponseSkills] = useState([]);

  // Occupations related state
  const [occupationsOptions] = useState([
    { value: 'occupation-1', label: 'Occupation 1' },
    { value: 'occupation-2', label: 'Occupation 2' },
    { value: 'occupation-3', label: 'Occupation 3' },
    { value: 'occupation-4', label: 'Occupation 4' },
    // Add more options as needed
  ]);
  const [selectedOccupations, setSelectedOccupations] = useState([]);
  const [suggestedOccupations, setSuggestedOccupations] = useState([]);
  const [responseOccupations, setResponseOccupations] = useState([]);

  const [activeTab, setActiveTab] = useState('skills');

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
      setTimeout(() => {
        const mockResponse = {
          suggestedSkills: ['skill-1', 'skill-2'], // only value possibilities from skillsOptions
          suggestedOccupations: ['occupation-1', 'occupation-2'] // only value possibilities from occupationsOptions
        };
        setResponseSkills(mockResponse.suggestedSkills); // Store the response from the server for skills
        setSuggestedSkills(mockResponse.suggestedSkills);
        setResponseOccupations(mockResponse.suggestedOccupations); // Store the response from the server for occupations
        setSuggestedOccupations(mockResponse.suggestedOccupations);
        setAnalyzed(true);
        setLoading(false);
      }, 2000);
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
    }
  };

  // Skills related functions
  const handleSkillsChange = (selected) => {
    const newSelectedSkills = selected.filter(option => !suggestedSkills.includes(option.value));
    setSelectedSkills(newSelectedSkills);
  };

  const addSkillFromSuggestion = (skillValue) => {
    const skill = skillsOptions.find(option => option.value === skillValue);
    if (skill) {
      setSelectedSkills([...selectedSkills, skill]);
      setSuggestedSkills(suggestedSkills.filter(s => s !== skillValue));
    }
  };

  const removeSkillTag = (option, isSkill) => {
    if (isSkill) {
      setSelectedSkills(selectedSkills.filter(item => item.value !== option.value));
    } else {
      setSuggestedSkills(suggestedSkills.filter(skill => skill !== option.value));
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

  // Occupations related functions
  const handleOccupationsChange = (selected) => {
    const newSelectedOccupations = selected.filter(option => !suggestedOccupations.includes(option.value));
    setSelectedOccupations(newSelectedOccupations);
  };

  const addOccupationFromSuggestion = (occupationValue) => {
    const occupation = occupationsOptions.find(option => option.value === occupationValue);
    if (occupation) {
      setSelectedOccupations([...selectedOccupations, occupation]);
      setSuggestedOccupations(suggestedOccupations.filter(o => o !== occupationValue));
    }
  };

  const removeOccupationTag = (option, isOccupation) => {
    if (isOccupation) {
      setSelectedOccupations(selectedOccupations.filter(item => item.value !== option.value));
    } else {
      setSuggestedOccupations(suggestedOccupations.filter(occupation => occupation !== option.value));
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

  const renderTags = (selectedOptions, isSkill) => {
    if (selectedOptions.length === 0) {
      return (
        <div className="text-gray-500 mb-4">
          {isSkill ? "No selections" : "No suggestions"}
        </div>
      );
    }

    return (
      <div className="flex gap-2 flex-wrap mb-4">
        {selectedOptions.map(option => (
          <Chip
            key={option.value}
            onClick={isSkill ? null : () => addSkillFromSuggestion(option.value)}
            onClose={() => removeSkillTag(option, isSkill)}
            className="flex items-center px-3 py-1 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow-sm hover:bg-gray-100 cursor-pointer"
          >
            {option.label}
          </Chip>
        ))}
      </div>
    );
  };

  const renderOccupationTags = (selectedOptions, isOccupation) => {
    if (selectedOptions.length === 0) {
      return (
        <div className="text-gray-500 mb-4">
          {isOccupation ? "No selections" : "No suggestions"}
        </div>
      );
    }

    return (
      <div className="flex gap-2 flex-wrap mb-4">
        {selectedOptions.map(option => (
          <Chip
            key={option.value}
            onClick={isOccupation ? null : () => addOccupationFromSuggestion(option.value)}
            onClose={() => removeOccupationTag(option, isOccupation)}
            className="flex items-center px-3 py-1 text-sm font-medium text-gray-800 bg-white border border-gray-300 rounded-full shadow-sm hover:bg-gray-100 cursor-pointer"
          >
            {option.label}
          </Chip>
        ))}
      </div>
    );
  };

  const handleTabChange = (tabKey) => {
    setActiveTab(tabKey);
  };

  return (
    <div className="min-h-screen">
      <Head>
        {/* TODO add title and logo */}
      </Head>

      <header className="flex justify-between items-center p-4 bg-white shadow">
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

      <main className="container mx-auto px-4 py-16">
        <h1 className="text-5xl font-bold text-center mb-4">CareerBot</h1>
        <p className="text-xl text-center mb-8">Analyze your resume and build your profile!</p>

        <div className="flex flex-col items-center">
          <div className="w-full max-w-md">
            <FileUpload onFileUpload={handleFileUpload} onFileRemove={handleFileRemove} />
          </div>

          {uploadedFilename && (
            <button
              className="w-full px-4 py-2 mt-4 text-white bg-blue-500 rounded shadow hover:bg-blue-600 max-w-md"
              onClick={handleSubmit}
              disabled={loading || analyzed}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          )}
        </div>

        {analyzed && (
          <div className="mt-16 selectionContainer">
            <h2 className="text-4xl font-semibold">Your profile</h2>
            <Tabs key="types" className="customTabs" selectedKey={activeTab} onSelectionChange={handleTabChange}>
              <Tab key="skills" title="Skills" className="customTab" />
              <Tab key="occupations" title="Occupations" className="customTab" />
            </Tabs>

            {activeTab === 'skills' && (
              <div className="skills-container w-full">
                <div className="suggestions-container bg-white shadow rounded-lg px-5 ">
                  <Card className='px-1'>
                    <CardHeader className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold mt-4">Bot suggestions</h3>
                      <Button auto flat onPress={resetSuggestedSkills}>Reset</Button>
                    </CardHeader>
                    <CardBody className="mt-4">
                      {renderTags(suggestedSkills.map(skillValue => {
                        const skill = skillsOptions.find(opt => opt.value === skillValue);
                        return skill ? { value: skill.value, label: skill.label } : null;
                      }).filter(skill => skill !== null), false)}
                    </CardBody>
                  </Card>
                </div>
                <hr className="my-4" />
                <div className="suggestions-container bg-white shadow rounded-lg px-5 ">
                  <Card className='px-1'>
                    <CardHeader className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold mt-4">Selections</h3>
                      <Button auto flat onPress={clearSelectedSkills}>Clear</Button>
                    </CardHeader>
                    <CardBody className="mt-4">
                      <div className="selections-container">
                        <SearchBar
                          options={skillsOptions.filter(opt => !selectedSkills.some(skill => skill.value === opt.value) && !suggestedSkills.includes(opt.value))}
                          placeholder="Select skills..."
                          onChange={handleSkillsChange}
                          value={selectedSkills}
                        />
                        <div>{renderTags(selectedSkills, true)}</div>
                      </div>
                    </CardBody>
                  </Card>
                </div>
              </div>
            )}

            {activeTab === 'occupations' && (
              <div className="occupations-container w-full">
                <div className="suggestions-container bg-white shadow rounded-lg px-5 ">
                  <Card className='px-1'>
                    <CardHeader className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold mt-4">Bot suggestions</h3>
                      <Button auto flat onPress={resetSuggestedOccupations}>Reset</Button>
                    </CardHeader>
                    <CardBody className="mt-4">
                      {renderOccupationTags(suggestedOccupations.map(occupationValue => {
                        const occupation = occupationsOptions.find(opt => opt.value === occupationValue);
                        return occupation ? { value: occupation.value, label: occupation.label } : null;
                      }).filter(occupation => occupation !== null), false)}
                    </CardBody>
                  </Card>
                </div>
                <hr className="my-4" />
                <div className="suggestions-container bg-white shadow rounded-lg px-5 ">
                  <Card className='px-1'>
                    <CardHeader className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold mt-4">Selections</h3>
                      <Button auto flat onPress={clearSelectedOccupations}>Clear</Button>
                    </CardHeader>
                    <CardBody className="mt-4">
                      <div className="selections-container">
                        <SearchBar
                          options={occupationsOptions.filter(opt => !selectedOccupations.some(occupation => occupation.value === opt.value) && !suggestedOccupations.includes(opt.value))}
                          placeholder="Select occupations..."
                          onChange={handleOccupationsChange}
                          value={selectedOccupations}
                        />
                        <div>{renderOccupationTags(selectedOccupations, true)}</div>
                      </div>
                    </CardBody>
                  </Card>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
