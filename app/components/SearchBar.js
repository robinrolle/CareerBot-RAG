import React, { useState } from 'react';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import { styled } from '@mui/material/styles';

const CustomTextField = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        borderRadius: '4px', // Ajuste cette valeur pour obtenir l'arrondi désiré
    },
});

const SearchBar = ({ options, value, onChange, placeholder }) => {
    const [inputValue, setInputValue] = useState('');

    const handleSelect = (event, newValue) => {
        if (newValue) {
            onChange([...value, newValue.value]);
            setInputValue('');  // Vider l'input après la sélection
        }
    };

    return (
        <Autocomplete
            options={options.filter(option => !value.includes(option.value))}
            getOptionLabel={(option) => option.label}
            onChange={handleSelect}
            inputValue={inputValue}
            onInputChange={(event, newInputValue) => {
                setInputValue(newInputValue);
            }}
            renderInput={(params) => (
                <CustomTextField
                    {...params}
                    variant="outlined"
                    placeholder={placeholder}
                />
            )}
            className="mb-4 w-full"
            disableClearable
            value={null} // Ajoutez ceci pour réinitialiser la sélection après le choix
        />
    );
};

export default SearchBar;
